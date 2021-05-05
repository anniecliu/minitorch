from project.datasets import Simple, Split, Xor

N = 100


def simple_classifier(x):
    "Classify with simple decision boundary based on position of x_0."
    if x[0] > 0.5:
        return 1.0
    else:
        return 0.0


def split_classifier(x):
    "Classify with split decision boundaries based on position of x_0."
    if x[0] < 0.2 or x[0] > 0.8:
        return 1.0
    else:
        return 0.0


def xor_classifier(x):
    "Classify strictly one-or-the-other decision boundaries based on both positions of x_0 and x_1."
    if (x[0] > 0.5 and x[1] < 0.5) or (x[0] < 0.5 and x[1] > 0.5):
        return 1.0
    else:
        return 0.0


Simple(N, vis=True).graph("Simple Classifier", model=simple_classifier)
Split(N, vis=True).graph("Split Classifier", model=split_classifier)
Xor(N, vis=True).graph("XOR Classifier", model=xor_classifier)
