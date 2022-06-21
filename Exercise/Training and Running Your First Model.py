"""
==============================
Preparing data
==============================
The first thing we do with a model is load data.

We have the sizes of boots and harnesses for 50 avalanche dogs.
We want to use harness size to estimate boot size. This means harness_size is our input. We want a model that will process the input and make its own estimations of the boot size (output).

"""
import pandas
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv
!pip install statsmodels


# Make a dictionary of data for boot sizes
# and harness size in cm
data = {
    'boot_size' : [ 39, 38, 37, 39, 38, 35, 37, 36, 35, 40, 
                    40, 36, 38, 39, 42, 42, 36, 36, 35, 41, 
                    42, 38, 37, 35, 40, 36, 35, 39, 41, 37, 
                    35, 41, 39, 41, 42, 42, 36, 37, 37, 39,
                    42, 35, 36, 41, 41, 41, 39, 39, 35, 39
 ],
    'harness_size': [ 58, 58, 52, 58, 57, 52, 55, 53, 49, 54,
                59, 56, 53, 58, 57, 58, 56, 51, 50, 59,
                59, 59, 55, 50, 55, 52, 53, 54, 61, 56,
                55, 60, 57, 56, 61, 58, 53, 57, 57, 55,
                60, 51, 52, 56, 55, 57, 58, 57, 51, 59
                ]
}

# Convert it into a table using pandas
dataset = pandas.DataFrame(data)

# Print the data
# In normal python we would write
# print(dataset)
# but in Jupyter notebooks, if we simple write the name
# of the variable and it is printed nicely 
dataset


"""
==============================
Selecting a model
==============================

The first thing we must do is select a model. We will start with a very simple model called OLS. This is just a straight line (sometimes called a trendline).

Let's use an existing library to create our model, but we won't train it yet
"""
# Load a library to do the hard work for us
import statsmodels.formula.api as smf

# First, we define our formula using a special syntax
# This says that boot_size is explained by harness_size
formula = "boot_size ~ harness_size"

# Create the model, but don't train it yet
model = smf.ols(formula = formula, data = dataset)

# Note that we have created our model but it does not 
# have internal parameters set yet
if not hasattr(model, 'params'):
    print("Model selected but it does not have parameters set. We need to train it!")
    


"""
==============================
Training our model
==============================

OLS models have two parameters (a slope and an offset), but these have not been set in our model yet. We need to train (fit) our model to find these values so that the model can reliably estimate dogs' boot size based on their harness size.

The code below fits our model to data you have now seen
"""

# Load some libraries to do the hard work for us
import graphing 

# Train (fit) the model so that it creates a line that 
# fits our data. This method does the hard work for
# us. We will look at how this method works in a later unit.
fitted_model = model.fit()

# Print information about our model now it has been fit
print("The following model parameters have been found:\n" +
        f"Line slope: {fitted_model.params[1]}\n"+
        f"Line Intercept: {fitted_model.params[0]}")

"""
The output should look like this:

The following model parameters have been found:
Line slope: 0.585925416738271
Line Intercept: 5.71910981268259

We could interpret these directly, but it's simpler to see it as a graph:
"""

import graphing

# Show a graph of the result
# Don't worry about how this works for now
graphing.scatter_2D(dataset,    label_x="harness_size", 
                                label_y="boot_size",
                                trendline=lambda x: fitted_model.params[1] * x + fitted_model.params[0]
                                )

"""
==============================
Using the model
==============================
The graph shows our original data as circles, with a red line through it. The red line shows our model.

We can look at this line to understand our model. For example, we can see that as harness size increases, so will the estimated boot size. For example, by looking at the red line, we can see that that a harness size of 52.5 (x axis) corresponds to a boot size of about 36.5 (y axis).

Now we've finished training, we can use our model to predict a dog's boot size from their harness size:
"""
# harness_size states the size of the harness we are interested in
harness_size = { 'harness_size' : [52.5] }

# Use the model to predict what size of boots the dog will fit
approximate_boot_size = fitted_model.predict(harness_size)

# Print the result
print("Estimated approximate_boot_size:")
print(approximate_boot_size[0])

"""
The output should look like this:

Estimated approximate_boot_size:
36.48019419144182
"""