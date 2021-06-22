# Controllable Summarization with Constrained Markov Decision Process

This repository contains the source code for our paper "Controllable Summarization with Constrained Markov Decision Process" to appear in TACL. 

**We will clean the code and release it soon.**

### Values of learned Lagrangian multipliers
The values of learned Lagrangian multipliers λ changes dynamically during training. In the following tables, we report the learned values of λ of our D.GPT2+CMDP model when the validation reward converges. 

**Length control:**
|                             | length bin constraint | 3-gram repetition constraint |
|-----------------------------|-----------------------|------------------------------|
| Values of learned λ   | 0.3312                | 0.3333                       |

**Entity control:**
|                             | QA constraint | entity repetition constraint | 3-gram repetition constraint |
|-----------------------------|-----------------------|------------------------------|------------------------------|
| Values of learned λ   | 0.1980                | 0.1810                       | 0.1972                       |

**Abstractiveness control:**
|                             | Abstractiveness bin constraint | conjunction constraint | 3-gram repetition constraint |
|-----------------------------|-----------------------|------------------------------|------------------------------|
| Values of learned λ (CNN/DM)   | 0.2842                | 0.1271                       | 0.2952                       |
| Values of learned λ (Newsroom-b)   | 0.4832                | 0.2210                       | 0.4898                       |
