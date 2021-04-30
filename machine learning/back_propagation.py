import math

# input
x1_apple_price = 100
x2_orange_price = 150

# weight initializer
w1_apple_num = 50
w2_orange_num = 15

# noise or weight
w3_tax = 1.1

# label
y_target_money = 715

learning_rate = 1e-5

d_w1 = 0
d_w2 = 0
d_w3 = 0
d_a3 = 0


def mse(z, y):
    return z - y


while True:
    input('Press Enter to continue \n')

    # propagation
    a1_apple_price_total = x1_apple_price * w1_apple_num
    a2_orange_price_total = x2_orange_price * w2_orange_num

    a3_hidden_layer = a1_apple_price_total + a2_orange_price_total

    z1_price_total = a3_hidden_layer * w3_tax

    # loss
    loss = mse(z1_price_total, y_target_money)
    print('prediction :', z1_price_total)
    print('loss :', loss)

    # back propagation
    d_w3 = loss * a3_hidden_layer
    print('d_w3 :', d_w3)
    d_a3 = loss * w3_tax
    d_w1 = d_a3 * x1_apple_price
    print('d_w1 :', d_w1)
    d_w2 = d_a3 * x2_orange_price
    print('d_w2 :', d_w2)

    # gradient descent
    w1_apple_num = w1_apple_num - (learning_rate * d_w1)
    print('new w1 :', w1_apple_num)
    w2_orange_num = w2_orange_num - (learning_rate * d_w2)
    print('new w2 :', w2_orange_num)
    # w3_tax = w3_tax - (learning_rate * d_w3)
    # print('new w3 :', w3_tax)
