import streamlit as st
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
import time

st.set_page_config(layout="wide")
st.title("Navier-Stokes, Diffusion Models, and MNIST Classification: Flow Animation")

st.markdown("""
This dashboard visualizes the deep parallels between **Navier-Stokes equations** (fluid flow), **diffusion models** (generative AI), and **MNIST classification**.
- **Blue background:** Probability "fluid" density
- **Colored arrows:** Drift/score field (Navier-Stokes velocity analog)
- **Black lines:** Classifier decision boundaries
- **Moving dot:** Your digit as a particle, flowing and stabilizing
---
""")

# --- Data and model setup ---
@st.cache_resource
def load_data_and_models():
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    images = mnist.data.float().view(-1, 28*28) / 255.0
    labels = mnist.targets
    images_2d = PCA(n_components=2).fit_transform(images[:2000])
    labels_2d = labels[:2000]
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    clf.fit(images_2d, labels_2d)
    return images, labels, images_2d, labels_2d, clf

images, labels, images_2d, labels_2d, clf = load_data_and_models()

T = 80
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise_seed=None):
    # Deterministic noise for reproducibility of the trajectory
    if noise_seed is not None:
        torch.manual_seed(noise_seed + t)
    noise = torch.randn_like(x0)
    return torch.sqrt(alphas_cumprod[t]) * x0 + torch.sqrt(1 - alphas_cumprod[t]) * noise

def estimate_score(samples, grid_x, grid_y):
    kde = gaussian_kde(samples.T)
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = kde(positions).reshape(grid_x.shape)
    grad_y, grad_x = np.gradient(np.log(density + 1e-8))
    return density, grad_x, grad_y

def classifier_decision(grid_points):
    probs = clf.predict_proba(grid_points)
    pred_class = np.argmax(probs, axis=1)
    return pred_class

# --- Sidebar controls ---
st.sidebar.header("Controls")
label = st.sidebar.selectbox("Digit to inject", list(range(10)), index=0)
steps = st.sidebar.slider("Animation steps", 20, 60, 40)
streamline_density = st.sidebar.slider("Streamline density", 0.5, 4.0, 2.0)
fade = st.sidebar.slider("Trail fade", 0.5, 0.98, 0.85)
speed = st.sidebar.slider("Animation speed (ms/frame)", 50, 400, 120)

# --- Choose a sample of the selected digit ---
indices = (labels_2d == label).nonzero(as_tuple=True)[0]
sample_idx = st.sidebar.selectbox("Sample index", list(range(len(indices))), 0)
chosen_idx = indices[sample_idx].item()
orig_point = torch.tensor(images_2d[chosen_idx], dtype=torch.float32)

# --- Simulate the trajectory for this particle ---
noise_seed = 42  # Ensures reproducible noise for the trajectory
particle_traj = []
for t in range(steps):
    p = q_sample(orig_point, t, noise_seed=noise_seed)
    particle_traj.append(p.numpy())

# --- For the background flow, use all data ---
all_traj = []
for t in range(steps):
    all_traj.append(q_sample(torch.tensor(images_2d, dtype=torch.float32), t).numpy())

# --- Grid for density/drift/decision ---
x_min, x_max = images_2d[:,0].min()-1, images_2d[:,0].max()+1
y_min, y_max = images_2d[:,1].min()-1, images_2d[:,1].max()+1
grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, 80),
    np.linspace(y_min, y_max, 80)
)
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
pred_class = classifier_decision(grid_points).reshape(grid_x.shape)

# --- Animation state ---
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

def next_frame():
    if st.session_state.frame < steps - 1:
        st.session_state.frame += 1

def prev_frame():
    if st.session_state.frame > 0:
        st.session_state.frame -= 1

def play():
    st.session_state.playing = True

def pause():
    st.session_state.playing = False

# --- Controls for animation ---
col1, col2, col3, col4 = st.columns([1,1,1,8])
with col1:
    st.button("◀️ Prev", on_click=prev_frame)
with col2:
    if not st.session_state.playing:
        st.button("▶️ Play", on_click=play)
    else:
        st.button("⏸ Pause", on_click=pause)
with col3:
    st.button("Next ▶️", on_click=next_frame)
with col4:
    st.slider("Step", 0, steps-1, st.session_state.frame, key="frame_slider")

# --- Animate if playing ---
if st.session_state.playing:
    time.sleep(speed/1000)
    if st.session_state.frame < steps - 1:
        st.session_state.frame += 1
        st.experimental_rerun()
    else:
        st.session_state.playing = False

# --- Draw the current frame ---
frame = st.session_state.frame
fig, ax = plt.subplots(figsize=(8,8))
density, grad_x, grad_y = estimate_score(all_traj[frame], grid_x, grid_y)
ax.imshow(density, extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.5, cmap='Blues')
ax.contour(grid_x, grid_y, pred_class, levels=np.arange(0,10), colors='k', linewidths=1, alpha=0.4)
ax.contourf(grid_x, grid_y, pred_class, levels=np.arange(0,11), alpha=0.09, cmap='tab10')
skip = (slice(None, None, 4), slice(None, None, 4))
mag = np.sqrt(grad_x**2 + grad_y**2)
ax.quiver(grid_x[skip], grid_y[skip], grad_x[skip], grad_y[skip], mag[skip],
          cmap='plasma', scale=40, width=0.008, alpha=0.7)
strm = ax.streamplot(grid_x, grid_y, grad_x, grad_y, color=mag, linewidth=1.5,
                     cmap='cool', density=streamline_density)
strm.lines.set_alpha(0.55)
# Particle trajectory (fading trail)
pts = np.array(particle_traj[:frame+1])
for j in range(1, len(pts)):
    ax.plot(pts[j-1:j+1,0], pts[j-1:j+1,1], color='red', alpha=fade**(len(pts)-j), linewidth=3)
ax.scatter(pts[-1,0], pts[-1,1], s=180, color='yellow', edgecolor='red', zorder=10, label="Particle")
pred = clf.predict([pts[-1]])
ax.text(pts[-1,0], pts[-1,1], str(pred[0]), fontsize=18, fontweight='bold', color='black', ha='center', va='center')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title(
    f"Step {frame}: Particle Flow, Drift, Decision Boundaries",
    fontsize=15, fontweight='bold'
)
ax.axis('off')
st.pyplot(fig)

st.markdown("""
---
**What are you seeing?**
- The **yellow dot** is your digit as a particle, moving through the probability flow.
- Its **fading red trail** shows its stochastic trajectory as it is denoised.
- The **blue background** is the evolving probability "fluid" density of MNIST digits.
- **Colored arrows** show the drift/score field (Navier-Stokes velocity analog).
- **Black lines** are classifier decision boundaries ("shorelines").
- The **digit label** shows the classifier's prediction as the particle stabilizes.
---
**Inspired by:** [Navier-Stokes and Diffusion Models: Theoretical Parallels](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/44523901/9ed6ce3e-45f0-42f9-8eef-47fe634804e9/Navier_Stokes_and_Fluidodynamics-1.pdf)
""")
