# gam_purification

Utilities for purifying Generalized Additive Models (GAMs) with interaction effects.

More details in [Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models](http://proceedings.mlr.press/v108/lengerich20a.html).

## Installing
```
pip install git+https://github.com/blengerich/gam_purification.git
```


## Citing
If you use these methods, please cite:
```
@InProceedings{pmlr-v108-lengerich20a,
  title = 	 {Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models},
  author =       {Lengerich, Benjamin and Tan, Sarah and Chang, Chun-Hao and Hooker, Giles and Caruana, Rich},
  booktitle = 	 {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2402--2412},
  year = 	 {2020},
  editor = 	 {Chiappa, Silvia and Calandra, Roberto},
  volume = 	 {108},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {26--28 Aug},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v108/lengerich20a/lengerich20a.pdf},
  url = 	 {https://proceedings.mlr.press/v108/lengerich20a.html},
  abstract = 	 {Models which estimate main effects of individual variables alongside interaction effects have an identifiability challenge: effects can be freely moved between main effects and interaction effects without changing the model prediction. This is a critical problem for interpretability because it permits “contradictory" models to represent the same function. To solve this problem, we propose pure interaction effects: variance in the outcome which cannot be represented by any subset of features. This definition has an equivalence with the Functional ANOVA decomposition. To compute this decomposition, we present a fast, exact algorithm that transforms any piecewise-constant function (such as a tree-based model) into a purified, canonical representation. We apply this algorithm to Generalized Additive Models with interactions trained on several datasets and show large disparity, including contradictions, between the apparent and the purified effects. These results underscore the need to specify data distributions and ensure identifiability before interpreting model parameters.}
}
```
