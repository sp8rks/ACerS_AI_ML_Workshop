# Problem-framing worksheet

Use this to check whether a problem from your own work is ready for machine learning. Be specific: vague answers here become failed projects later.

| Question | Your answer |
|---|---|
| **Decision.** What will you do differently based on the model's output? | |
| **Output.** Which property or label, in what units? | |
| **Inputs.** Which variables will you *actually have* at prediction time (composition, processing, structure, images…)? | |
| **Data volume.** How many labeled examples exist today? Where do they come from? | |
| **Label quality.** How noisy are the labels (scatter in repeat measurements)? Are they defined the same way across sources? | |
| **Baseline.** What does an expert, a simple rule or a physics model achieve today? | |
| **Success.** Which metric, and what error would still be *useful*? | |
| **Interpolation or extrapolation?** Will you predict inside or outside the composition/process range of your data? | |
| **Cost of being wrong.** What happens if the model is wrong? How will you catch it? | |
| **Next data.** Can you generate more data if the model is uncertain (experiments, simulations)? | |

## Red flags 🚩

- Fewer than ~50 examples and no physics-informed features
- Inputs that will not be known when you need the prediction
- A target measured differently across labs or papers
- "We want to predict everything"
- The decision would be the same whatever the model says

## Green flags ✅

- A clear decision, a cheap way to verify predictions, and a loop to add new data
- Domain knowledge you can turn into features or constraints
- A measurable baseline the model has to beat
