import numpy as np
import matplotlib.pyplot as plt
import h5py


# input the learning rate, and give a proper limitation of
# the input value
print("Please define the learning rate.")
learning_rate = float(input('>'))
if learning_rate > 0.015:
    raise Exception(
        'This learning_rate might be to large. 0.0075 is suggested.')
if learning_rate < 0.0003:
    raise Exception(
        'This learning_rate might be to small. 0.0075 is suggested.')
# input the number of iterations, also give a limitation
print('Please define the iteration times.')
iteration_times = int(input('>'))
if iteration_times < 1500:
    raise Exception(
        'This iteration times are not enough.')
if iteration_times > 6000:
    raise Exception(
        'Be kind to your PC, it will get tired for so many iterations.')
# choose an image from training set to know what is loading
print('Please choose an image from training set to view.')
image_number_for_training = int(input('>'))
if image_number_for_training < 0:
    raise Exception('Negative value is invalid.')
if image_number_for_training >= 209:
    raise Exception('Sorry, we only have 209 image in training set.')
# user can check the classification result of testing set, choose one!
print('Which image in test set you want check?')
image_number_for_testing = int(input('>'))
if image_number_for_testing < 0:
    raise Exception('At least it should start from 0 right.')
if image_number_for_testing >= 50:
    raise Exception('Sorry, we only have 50 images for testing.')


def load_train_set(filename):
    '''
    This function will load the train set for the classifier, which is h5 file.

    **Parameters**
        filename: *str*
            the file need to be loaded.

    **Return**
        train_image: *array*
            an array with dimension of (piexl, piexl, number of image).
            Indeed is our train set of images.
        train_label: *array*
            an array with dimension of (1, number of image), reflecting
            is or is not a cat image.
    '''
    if filename != 'train_cat.h5':
        raise Exception('a wrong file has been loaded')
    train_dataset = h5py.File(filename, 'r')
    # read image and labels seperately.
    train_image_orig = np.array(train_dataset['train_set_x'][:])
    train_label_orig = np.array(train_dataset['train_set_y'][:])
    train_label = train_label_orig.reshape((1, train_label_orig.shape[0]))
    # reshape the array so that later when doing the mechine learing,
    # it will have the desired dimension.
    # standardlize the image piexls by dividing 255 which is maximum RGB value.
    train_image = train_image_orig.reshape(
        (train_image_orig.shape[0], -1)).T / 255
    print("dimension of training image before transformation to 1-D is: " +
          str(train_image_orig.shape))
    print("dimension of training label is:  : " + str(train_label.shape))
    return train_image, train_label


def load_test_set(filename):
    '''
    This function will load the test set for the classifier, which is h5 file.

    **Parameters**
        filename: *str*
            the file need to be loaded.

    **Return**
        test_image: *array*
            an array with dimension of (piexl, piexl, number of image).
            Indeed is our test set of images.
        test_label: *array*
            an array with dimension of (1, number of image), reflecting
            is or is not a cat image.
    '''
    if filename != 'test_cat.h5':
        raise Exception('a wrong file has been loaded')
    test_dataset = h5py.File(filename, "r")
    test_image_orig = np.array(test_dataset['test_set_x'][:])
    test_label_orig = np.array(test_dataset['test_set_y'][:])
    test_label = test_label_orig.reshape((1, test_label_orig.shape[0]))
    test_image = test_image_orig.reshape(
        (test_image_orig.shape[0], -1)).T / 255
    print('dimension of testing image before transformation to 1-D is: ' +
          str(test_image_orig.shape))
    print('dimension of testing label is: ' + str(test_label.shape))
    return test_image, test_label


def check_input_image(imagelabel, filename):
    '''
    This function will be used to pick a image from h5.file and plot.
    In a word, a check method for the classifier.

    **Parameters**
        imagelabel: *int*
            choose which image the user want to check.
        filename: *str*
            choose the image in which image set the user
            want to check.

    **Return**
        None
    '''
    which_file = h5py.File(filename, 'r')
    if filename == 'train_cat.h5':
        train_set_x_orig = np.array(which_file["train_set_x"][:])
        which_image = train_set_x_orig[imagelabel]
    elif filename == 'test_cat.h5':
        raise Exception('You need to check within the training file.')
    plt.imshow(which_image)
    plt.ylabel('pixels')
    plt.xlabel('pixels')
    plt.title('A simple unit testing')
    plt.show()
    return


class function_to_squeeze_activation(object):
    '''
    This class will enable user to choose one method
    to squeeze the activation value.
    '''

    def __init__(self, input):
        # initializing the class.
        self.input = input
        return

    def __relu__(self):
        # the relu function
        result = np.maximum(0, self.input)
        return result

    def __reluderi__(self):
        # derivative of relu
        result = np.maximum(0, self.input / np.abs(self.input))
        if np.abs(self.input).any == 0:
            raise Exception(
                'Please choose another function to squeeze acitivation.')
        return result

    def __sigmoid__(self):
        # the simgoid function.
        result = 1 / (1 + np.exp(-self.input))
        if (1 + np.exp(-self.input)).any == 0:
            raise Exception(
                'Please choose another function to squeeze acitivation.')
        return result

    def __sigmoidderi__(self):
        # derivative of sigmoid
        result = self.input * (1 - self.input)
        return result

    def __tanh__(self):
        # the other one function, which was not chosen to use
        e1 = np.exp(self.input)
        e2 = np.exp(-self.input)
        result = (e1 - e2) / (e1 + e2)
        if (e1 + e2).any == 0:
            raise Exception(
                'Please choose another function to squeeze acitivation.')
        return result

    def __tanhderi__(self):
        # derivative of this function
        result = 1 - a ** 2
        return result


def forward_propagation(weight1, bias1, weight2, bias2, image_set, is_cat):
    '''
    This function will calculate the cost of the learning.

    **Parameters**
        weight1: *array*
            a given array used to adjust the activation
            value at hidden layer.
        bias1: *array*
            an array to forward adjust the activation
            value at hidden layer.
        weight2: *array*
            a given array used to adjust the activation value.
        bias2: *array*
            an array to forward adjust the activation value.
        image_set: *array*
            an array has been transformed into dimension
            (piexl * piexl * 3, 1).
        is_cat: *array*
            the return value indicating the image is or is not a cat
            of function load_train_set

    **Return**
        cost: *int*
            tells how clever the computer is.
    '''
    # ensure the the type is array.
    weight1 = np.array(weight1)
    weight2 = np.array(weight2)
    # calculate the activation value for the hidden layer.
    weighted_and_biased_activation1 = np.array(
        np.dot(weight1, image_set) + bias1)
    # fix the activation value for hidden layer within [0, 1]
    # One thing to alert, if use sigmoid or tanh function
    # here. It will give a overflow error.
    new_activation1 = function_to_squeeze_activation(
        weighted_and_biased_activation1).__relu__()
    # calculate the activation value for the last layer
    weighted_and_biased_activation2 = np.array(
        np.dot(weight2, new_activation1) + bias2)
    # fix the activation value for last layer within [0, 1]
    new_activation2 = function_to_squeeze_activation(
        weighted_and_biased_activation2).__sigmoid__()
    # calculate the cost
    cost = (- 1 / image_set.shape[1]) * np.sum(is_cat * np.log(
        new_activation2) + (1 - is_cat) * np.log(1 - new_activation2))
    return cost


def find_del(weight1, bias1, weight2, bias2, image_set, is_cat):
    '''
    This function will calculate the del of weight and bias.

    **Parameters**
        weight1: *array*
            a given array used to adjust the activation value at hidden layer.
        bias1: *array*
            an array to forward adjust the activation value at hidden layer.
        weight2: *array*
            a given array used to adjust the activation value.
        bias2: *array*
            an array to forward adjust the activation value.
        image_set: *array*
            an array has been transformed into dimension
            (piexl * piexl * 3, 1).
        is_cat: *array*
            the return value indicating the image is or is not a cat
            of function load_train_set

    **Return**
        dw2: *array*
            tells the efficient learning direction of last layer weight.
        db2: *array*
            tells the efficient learning direction of last layer bias.
        dw1: *array*
            tells the efficient learning direction of hidden layer weight.
        db1: *array*
            tells the efficient learning direction of hidden layer bias.
    '''
    weight1 = np.array(weight1)
    weight2 = np.array(weight2)
    weighted_and_biased_activation1 = np.dot(
        weight1, image_set) + bias1
    new_activation1 = function_to_squeeze_activation(
        weighted_and_biased_activation1).__relu__()
    weighted_and_biased_activation2 = np.dot(
        weight2, new_activation1) + bias2
    new_activation2 = function_to_squeeze_activation(
        weighted_and_biased_activation2).__sigmoid__()
    # calculate the del of weight and bias to achieve the
    # gradient desent for this project. As del of
    # activation values correlated with del of weight and bias.
    # They also need to be calculated.
    da2 = -1 * (np.divide(is_cat, new_activation2) -
                np.divide(1 - is_cat, 1 - new_activation2))
    dz2 = da2 * \
        function_to_squeeze_activation(new_activation2).__sigmoidderi__()
    da1 = np.dot(weight2.T, dz2)
    dz1 = da1 * \
        function_to_squeeze_activation(
            weighted_and_biased_activation1).__reluderi__()
    # now, obtain the del for weight and bias
    dw2 = 1 / image_set.shape[1] * np.dot(dz2, new_activation1.T)
    db2 = 1 / image_set.shape[1] * \
        np.sum(dz2, axis=1, keepdims=True)
    dw1 = 1 / image_set.shape[1] * \
        np.dot(dz1, image_set.T)
    db1 = 1 / image_set.shape[1] * \
        np.sum(dz1, axis=1, keepdims=True)
    return dw2, db2, dw1, db1


def backward(image_set, is_cat_or_not, w1, w2, b1, b2, learning_rate):
    '''
    This function will claculate cost, and renew the value of weight and bias.
    To be simple, achieve the gradient desent.

    **Parameters**
        image_set: *array*
            an array has been transformed into dimension
            (piexl * piexl * 3, 1).
        is_cat_or_not: *array*
            the return value indicating the image is or is not a cat
            of function load_train_set
        w1: *array*
            a given array used to adjust the activation value at hidden layer.
        w2: *array*
            a given array used to adjust the activation value.
        b1: *array*
            an array to forward adjust the activation value at hidden layer.
        b2: *array*
            an array to forward adjust the activation value.
        learning rate: *int*
            it tells how much would the weight and bias get optimized for
            each iteration

    **Return**
        w2: *array*
            the opitimized last layer weight after one iteration
        b2: *array*
            the opitimized last layer bias after one iteration
        w1: *array*
            the opitimized hidden layer weight after one iteration
        b1: *array*
            the opitimized hidden layer bias after one iteration
        cost: *int*
            the opitimized cost after one iteration
    '''
    cost = forward_propagation(
        w1, b1, w2, b2, image_set, is_cat_or_not)
    dw2, db2, dw1, db1 = find_del(
        w1, b1, w2, b2, image_set, is_cat_or_not)
    # optimize the weight and bias
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    return w1, w2, b1, b2, cost


def discern_imgae_is_or_not_cat(weight1, bias1, weight2, bias2, image_set):
    '''
    This function will use the optimized weight and bias after
    learning to discern the image is or not a cat, to classify.

    **Parameters**
        weight1: *array*
            a given array used to adjust the activation value at hidden layer.
        bias1: *array*
            an array to forward adjust the activation value at hidden layer.
        weight2: *array*
            a given array used to adjust the activation value.
        bias2: *array*
            an array to forward adjust the activation value.
        image_set: *array*
            the test set of image.

    **Returns**
        discren_result: *array*
            an array made up by 0 and 1 to classify the input images.
    '''
    weight2 = np.array(weight2)
    weight1 = np.array(weight1)
    # initialize a zero array, which will be used to record
    # the classification result.
    discren_result = np.zeros((1, image_set.shape[1]))
    weighted_and_biased_activation1 = np.dot(
        weight1, image_set) + bias1
    new_activation1 = function_to_squeeze_activation(
        weighted_and_biased_activation1).__relu__()
    weighted_and_biased_activation2 = np.dot(weight2, new_activation1) + bias2
    new_activation2 = function_to_squeeze_activation(
        weighted_and_biased_activation2).__sigmoid__()
    # update the value (zeros) in the initialized array, we
    # assume if the computer say it is 50% a cat, then it is
    # a cat.
    for i in range(0, new_activation2.shape[1]):
        if new_activation2[0, i] > 0.5:
            discren_result[0, i] = 1
        else:
            discren_result[0, i] = 0
    return discren_result


def train_precision(w1, w2, b1, b2, transformed_image_set, is_cat_or_not):
    '''
    This function will tell user the accuracy of the training image set
    comparaing to the label from h5.file.

    **Parameters**
        w1: *array*
            a given array used to adjust the activation value at hidden layer.
        w2: *array*
            a given array used to adjust the activation value.
        b1: *array*
            an array to forward adjust the activation value at hidden layer.
        b2: *array*
            an array to forward adjust the activation value.

    **Returns**
        None
    '''
    y = discern_imgae_is_or_not_cat(
        w1, b1, w2, b2, transformed_image_set)
    # take an average value of discrepancy as the accuracy
    lop = 100 * (1 - np.mean(np.abs(y - is_cat_or_not)))
    print("Accuracy of training set is： " + str(lop) + '%')
    return


def test_precision(w1, w2, b1, b2, transformed_image_set, is_cat_or_not):
    '''
    This function will tell user the accuracy of the testing image set
    comparaing to the label from h5.file.

    **Parameters**
        w1: *array*
            a given array used to adjust the activation value at hidden layer.
        w2: *array*
            a given array used to adjust the activation value.
        b1: *array*
            an array to forward adjust the activation value at hidden layer.
        b2: *array*
            an array to forward adjust the activation value.

    **Returns**
        None
    '''
    y = discern_imgae_is_or_not_cat(
        w1, b1, w2, b2, transformed_image_set)
    lop = 100 * (1 - np.mean(np.abs(y - is_cat_or_not)))
    print('Accuracy of testing set is： ' + str(lop) + '%')
    return


def classify_result(weight1, bias1, weight2, bias2, which_image, image_set):
     '''
    This function will pick one image to show with the classified result

    **Parameters**
        weight1: *array*
            a given array used to adjust the activation value at hidden layer.
        bias1: *array*
            an array to forward adjust the activation value at hidden layer.
        weight2: *array*
            a given array used to adjust the activation value.
        bias2: *array*
            an array to forward adjust the activation value.
        which_image: *int*
            the order of the picture to be reviewed in the image_set
        image_set: *array*
            the test set of image.

    **Returns**
        None
    '''
    the_classified_array = discern_imgae_is_or_not_cat(
        weight1, bias1, weight2, bias2, image_set)
    yes_or_no = np.squeeze(the_classified_array)
    yes_or_no = yes_or_no.tolist()
    print(yes_or_no[which_image - 1])
    if yes_or_no[which_image - 1] == 0.0:
        which_file = h5py.File('test_cat.h5', 'r')
        train_set_x_orig = np.array(which_file['test_set_x'][:])
        which_image_to_show = train_set_x_orig[which_image]
        plt.imshow(which_image_to_show)
        plt.xlabel('This is not a cat!')
        plt.show()
    else:
        which_file = h5py.File('test_cat.h5', 'r')
        train_set_x_orig = np.array(which_file['test_set_x'][:])
        which_image_to_show = train_set_x_orig[which_image]
        plt.imshow(which_image_to_show)
        plt.xlabel('This is a cat!')
        plt.show()
    return


if __name__ == '__main__':
    # please just use this seed to avoid possible overflow problem.
    np.random.seed(1)
    # initializing the data from h5.file.
    train_set_x, train_set_y = load_train_set('train_cat.h5')
    test_set_x, test_set_y = load_test_set('test_cat.h5')
    # hidden layer neuron number is 4, which is dim here
    n, m, dim = train_set_x.shape[1], train_set_x.shape[0], 4
    # initializing weight and bias
    w1 = np.random.randn(dim, m) * 0.001
    w2 = np.random.randn(1, dim) * 0.001
    b1 = np.zeros((dim, 1), dtype='float')
    b2 = np.zeros((1, 1), dtype='float')
    # a simple unit test
    check_input_image(image_number_for_training, 'train_cat.h5')
    cost_array = []
    x_axis = []
    # recording the variation in cost during gradient desent
    for i in range(iteration_times):
        w1, w2, b1, b2, cost = backward(
            train_set_x, train_set_y, w1, w2, b1, b2, learning_rate)
        if i % 500 == 0:
            print('The cost for ' + str(i) + 'times iteration is ' + str(cost))
            cost_array.append(cost)
            x_axis.append(i)
    # If user want to view the optimized weight and bias, please uncomment
    # the following two rows.
    # print('The optimized weight and bias are: weight1 = ' + str(w1) +
    #      'weight2 = ' + str(w2) + 'bias1 = ' + str(b1) + 'bias2 = ' + str(b2))
    # calculate the precision of training set and testing set
    # with the optimized weight and bias
    train_precision(w1, w2, b1, b2, train_set_x, train_set_y)
    test_precision(w1, w2, b1, b2, test_set_x, test_set_y)
    standard_cost_array = []
    # set up a comparision with same iteration time
    # but differernt learning rate
    w1 = np.random.randn(dim, m) * 0.001
    w2 = np.random.randn(1, dim) * 0.001
    b1 = np.zeros((dim, 1), dtype='float')
    b2 = np.zeros((1, 1), dtype='float')
    for j in range(iteration_times):
        w1, w2, b1, b2, cost = backward(
            train_set_x, train_set_y, w1, w2, b1, b2, 0.0075)
        if j % 500 == 0:
            standard_cost_array.append(cost)
    # plot both curves
    plt.plot(x_axis, standard_cost_array, color='green',
             label='learning rate = 0.0075')
    plt.plot(x_axis, cost_array, color='red',
             label='learning rate = ' + str(learning_rate))
    plt.legend()
    plt.ylabel('Cost')
    plt.xlabel('Number of training rounds')
    plt.title('learning_rate =' + str(learning_rate))
    plt.show()
    # To have a look at the image from testing set and check the classification
    # result.
    classify_result(
        w1, b1, w2, b2, image_number_for_testing, train_set_x)
