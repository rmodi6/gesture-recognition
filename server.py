'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json

# IMPORTS
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import interp1d
import numpy as np

app = Flask(__name__)

# PRE-PROCESSING
# Number of sample points
num_sample_points = 100
# Calculate 100 evenly spaced numbers between 0 and 1
evenly_spaced_100_numbers = np.linspace(0, 1, num_sample_points)
# Calculate alphas for location score
alphas = np.zeros((num_sample_points))
mid_point = num_sample_points // 2
for i in range(mid_point):
    x = i/2450
    alphas[mid_point - i - 1], alphas[mid_point + i] = x, x

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)

    # Calculate the euclidean distance between consecutive points
    distance = np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2)
    # Calculate the cumulative distance
    cumulative_distance = np.cumsum(distance)
    # Normalize the cumulative distance between 0 and 1
    total_distance = cumulative_distance[-1]
    cumulative_distance_norm = cumulative_distance / total_distance

    # Interpolate numbers into 1-D space for both X and Y
    interp1d_X = interp1d(cumulative_distance_norm, points_X, kind='linear')
    interp1d_Y = interp1d(cumulative_distance_norm, points_Y, kind='linear')

    # Create the sample points for X and Y
    sample_points_X, sample_points_Y = interp1d_X(evenly_spaced_100_numbers), interp1d_Y(evenly_spaced_100_numbers)
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

# Normalize every template
L = 200
templates_width = np.max(template_sample_points_X, axis=1) - np.min(template_sample_points_X, axis=1)
templates_height = np.max(template_sample_points_Y, axis=1) - np.min(template_sample_points_Y, axis=1)
s = L / np.maximum(1, np.max(np.array([templates_width, templates_height]), axis=0))

scaling_matrix = np.diag(s)
scaled_template_points_X = np.matmul(scaling_matrix, template_sample_points_X)
scaled_template_points_Y = np.matmul(scaling_matrix, template_sample_points_Y)

scaled_template_centroid_X, scaled_template_centroid_Y = np.mean(scaled_template_points_X, axis=1), np.mean(scaled_template_points_Y, axis=1)

tx, ty = 0 - scaled_template_centroid_X, 0 - scaled_template_centroid_Y
translation_matrix_X = np.reshape(tx, (-1, 1))
translation_matrix_Y = np.reshape(ty, (-1, 1))
normalized_template_sample_points_X = translation_matrix_X + scaled_template_points_X
normalized_template_sample_points_Y = translation_matrix_Y + scaled_template_points_Y


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 30
    # TODO: Do pruning (12 points)

    # Create numpy array for gesture start and end point [[x, y]]
    gesture_start_point = np.array([gesture_points_X[0], gesture_points_Y[0]])
    gesture_end_point = np.array([gesture_points_X[-1], gesture_points_Y[-1]])

    # Number of templates
    num_templates = len(template_sample_points_X)
    # Gather the start points and end points of templates in a numpy matrix [[x1, y1], [x2, y2], ..., [xn, yn]]
    template_start_points = np.array([[template_sample_points_X[i][0], template_sample_points_Y[i][0]] for i in range(num_templates)])
    template_end_points = np.array([[template_sample_points_X[i][-1], template_sample_points_Y[i][-1]] for i in range(num_templates)])

    # Calculate distances between start points of gesture and templates and end points of gesture and templates
    start_distances = euclidean_distances(np.reshape(gesture_start_point, (1, -1)), template_start_points)[0]
    end_distances = euclidean_distances(np.reshape(gesture_end_point, (1, -1)), template_end_points)[0]

    # Get indices whose start + end distances are less than the threshold
    valid_indices = np.where((start_distances + end_distances) < threshold)[0]

    # Gather valid template sample points and valid words using the valid indices
    valid_template_sample_points_X = np.array(template_sample_points_X)[valid_indices]
    valid_template_sample_points_Y = np.array(template_sample_points_Y)[valid_indices]
    valid_words = [words[valid_index] for valid_index in valid_indices]

    return valid_indices, valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(valid_indices, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 200
    gesture_width = np.max(gesture_sample_points_X) - np.min(gesture_sample_points_X)
    gesture_height = np.max(gesture_sample_points_Y) - np.min(gesture_sample_points_Y)
    s = L / max(gesture_width, gesture_height, 1)

    scaling_matrix = np.array([[s, 0],
                               [0, s]])
    old_gesture_points = np.array([gesture_sample_points_X,
                                   gesture_sample_points_Y])
    scaled_gesture_points = np.matmul(scaling_matrix, old_gesture_points)

    scaled_gesture_centroid_X, scaled_gesture_centroid_Y = np.mean(scaled_gesture_points[0]), np.mean(scaled_gesture_points[1])

    tx, ty = 0 - scaled_gesture_centroid_X, 0 - scaled_gesture_centroid_Y
    translation_matrix = np.array([[tx],
                                   [ty]])
    normalized_gesture_sample_points = translation_matrix + scaled_gesture_points

    # TODO: Calculate shape scores (12 points)

    valid_normalized_template_sample_points_X = normalized_template_sample_points_X[valid_indices]
    valid_normalized_template_sample_points_Y = normalized_template_sample_points_Y[valid_indices]

    x_ = (valid_normalized_template_sample_points_X - np.reshape(normalized_gesture_sample_points[0], (1, -1))) ** 2
    y_ = (valid_normalized_template_sample_points_Y - np.reshape(normalized_gesture_sample_points[1], (1, -1))) ** 2
    distances = (x_ + y_) ** 0.5
    shape_scores = np.sum(distances, axis=1) / num_sample_points

    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores (12 points)

    # Initialize location scores
    location_scores = np.zeros((len(valid_template_sample_points_X)))
    # Create a list of gesture points [[xi, yi]]
    gesture_points = [[gesture_sample_points_X[j], gesture_sample_points_Y[j]] for j in range(num_sample_points)]

    # For each template
    for i in range(len(valid_template_sample_points_X)):
        # Create a list of template points
        template_points = [[valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]] for j in range(num_sample_points)]
        # Calculate distance of each gesture point with each template point
        distances = euclidean_distances(gesture_points, template_points)
        # Find the distance of the closest gesture point to each template point
        template_gesture_min_distances = np.min(distances, axis=0)
        # Find the distance of the closest template point to each gesture point
        gesture_template_min_distances = np.min(distances, axis=1)
        # If any gesture point is not within the radius tunnel or any template point is not within the radius tunnel
        if np.any(gesture_template_min_distances > radius) or np.any(template_gesture_min_distances > radius):
            # Calculate delta as the distance of each gesture point with corresponding template point
            deltas = np.diagonal(distances)
            # Calculate location score as sum of product of alpha and delta for each point
            location_scores[i] = np.sum(np.multiply(alphas, deltas))

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.1
    # TODO: Set your own location weight
    location_coef = 1 - shape_coef
    integration_scores = shape_coef * shape_scores + location_coef * location_scores
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    min_score = np.min(np.array(integration_scores))
    min_score_indices = np.where(integration_scores == min_score)[0]
    best_words = [valid_words[min_score_index] for min_score_index in min_score_indices]
    return ' '.join(best_words)


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    # gesture_points_X = [gesture_points_X]
    # gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_indices, valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(valid_indices, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
