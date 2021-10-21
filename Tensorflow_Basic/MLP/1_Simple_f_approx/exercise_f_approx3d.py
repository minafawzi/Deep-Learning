# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# It is always good practice to set the hyper-parameters at the beginning of the script
# And even better to define a params class if the script is long and complex
LR = 0.05
N_NEURONS = 100
N_EPOCHS = 10000

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# Gets the input interval and the targets
x1 = np.linspace(-3, 3, 500)
x2 = np.linspace(-3, 3, 500)

# reshape the inputs to work with the MLP
p= np.column_stack((x1,x2))
#print(p)

#Define the function
t = x1*x1 - x2*x2

# %% -------------------------------------- Training Prep --------------------------------------------------------------
# Defines the model

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(2,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Prepares a Stochastic Gradient Descent optimizer and a Mean Squared Error performance index
#model.compile(optimizer=Adam(), loss="mean_squared_error")
model.compile(optimizer=SGD(lr=LR), loss="mean_squared_error")

# %% -------------------------------------- Training Loop --------------------------------------------------------------
# Trains the model. We use full Batch GD.
train_hist = model.fit(p, t, epochs=N_EPOCHS, batch_size=len(p))

# %% --------------------------------------  meshgrid reshaping ---------------------------------------------------------------
# reate meshgrid on the original function
X1, X2 = np.meshgrid(x1, x2)
T = X1*X1 - X2*X2

#print(T.shape)
#reshape the meshgrid to fit NN
X3 = X1.reshape(-1,1)
X4 = X2.reshape(-1,1)

#Reshap the input by stacking the two X3 and X4 columns
P= np.column_stack((X3,X4))

#
Output = model.predict(P)
Output2 = Output.reshape(T.shape)
#print(Output.shape)
#print(X3.shape)
#print(P.shape)
plt.title("MLP fit to x1^2-x2^2 | MSE: {:.5f}".format(train_hist.history["loss"][-1]))
##
#print("X1:")
#print(X1)
#print("X2:")
#print(X2)
#print("T:")
#print(T)

#print("X3: ")
#print(X3)
#print("X4: ")
#print(X4)
#print("P: ")
#print(P)
#print("Output: ")
#print(Output)
#print("Output2: ")
#print(Output2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, Output2, 50, cmap='Reds')
ax.contour3D(X1, X2, T, 50, cmap='Wistia')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y');
plt.legend()
plt.show()

# %% -------------------------------------------------------------------------------------------------------------------

# -------------------------------------
# Approximate a 3D Function using a MLP
# -------------------------------------

# 1. Define the function y = x1**2 - x2**2 you will train the MLP to approximate. A 3D plot can be found at:
# # http://www.livephysics.com/tools/mathematical-tools/online-3-d-function-grapher/

# 2. Define a helper function to plot the real function and the MLP approximation. Hint:
# from mpl_toolkits.mplot3d import Axes3D, use ax.contour3D on 3 inputs with shapes (sqrt(n_examples), sqrt(n_examples))
# You may do 3. first to get the data and figure out why the shapes are like this

# 3. Generate the data to train the network using the function you defined in 1. Hint:
# Use np.meshgrid() and then reshape the input to (n_examples, 2) and the target to (n_examples, 1)

# 4. Define a MLP to approximate this function using the data you just generated.
# Play with the number of layers, neurons and hidden activation functions (tanh, ReLu, etc.)

# 5. Use Adam or another optimizer and train the network. Find an appropriate learning rate and number of epochs.

# 6. Use the function you defined in 2. to visualize how well your MLP fits the original function

# %% -------------------------------------------------------------------------------------------------------------------
