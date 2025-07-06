---
layout: post
title: "Lecture Notes: Navier-Stokes Equations and Their Relation to LLM Diffusion Models"
author: Francesco Orsi
date: {{ page.date | date_to_long_string }}
---

## Abstract

These lecture notes provide a comprehensive overview of the Navier-Stokes Equations (NSE), fundamental to fluid dynamics, and explore their surprising conceptual and mathematical parallels with modern Large Language Model (LLM) diffusion models. We delve into the derivation and physical interpretation of the NSE, discussing their complexities and the ongoing millennium prize problem. Subsequently, we introduce the theoretical underpinnings of diffusion models, focusing on stochastic differential equations (SDEs) and their role in generative AI. The core of these notes lies in drawing analogies between the flow dynamics described by NSE and the probabilistic flow in diffusion models, highlighting how concepts like conservation laws and transport phenomena find echoes in the denoising process of generative models.

## Table of Contents
- [Introduction to Navier-Stokes Equations](#introduction-to-navier-stokes-equations)
- [Formulation of the Navier-Stokes Equations](#formulation-of-the-navier-stokes-equations)
- [Properties and Challenges of NSE](#properties-and-challenges-of-nse)
- [Introduction to Diffusion Models in Machine Learning](#introduction-to-diffusion-models-in-machine-learning)
- [LLM Diffusion Models](#llm-diffusion-models)
- [The Relation Between Navier-Stokes Equations and Diffusion Models](#the-relation-between-navier-stokes-equations-and-diffusion-models)
- [Conclusion](#conclusion)

## Introduction to Navier-Stokes Equations

The Navier-Stokes Equations (NSE) are a set of partial differential equations (PDEs) that describe the motion of viscous fluid substances.

### Historical Context

The equations were independently derived by Claude-Louis Navier in 1822 and George Gabriel Stokes in 1845.

### The Millennium Prize Problem

One of the most significant challenges in mathematics is proving the existence and smoothness of solutions to the NSE for three-dimensional incompressible flows. This is one of the seven Millennium Prize Problems.

## Formulation of the Navier-Stokes Equations

### Conservation of Mass (Continuity Equation)

For an incompressible fluid, the density \( \rho \) is constant. The continuity equation states:

```math
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0
