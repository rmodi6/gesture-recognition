Pruning:
=======
A slight modification in pruning done is that, instead of checking the threshold for either start or end distance,
I have calculated the threshold limit on the sum of start point and end point distances. This allows for slight errors
in gestures at the beginning or end improving accuracy for mistyped gestures.

Code Modifications:
==================
- I have commented out lines 332 and 333 which cause errors in sampling points for gestures.
- Added a check to see if there are no valid words remaining after pruning, then return "Word not found"
- I have pre-calculated the normalized template sample points for improving running time.
- I have also pre-calculated alphas used in location scores for improving running time.

References:
==========
- sampling equidistant points on a line: https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
- euclidean distance between a vector and matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
- scaling: https://www.gatevidyalay.com/scaling-in-computer-graphics-definition-examples/
- translation: https://www.gatevidyalay.com/2d-transformation-in-computer-graphics-translation-examples/