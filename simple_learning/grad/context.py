class Context:
    """Context behind every Tensor's creation. Holds the necessary data for backprop."""
    def __init__(self, function, parents=None, saved_data=None):
        self.function = function
        self.parents = parents

        if saved_data is None:
            self.saved_data = []

    def save_for_backward(self, *args):
        self.saved_data.extend(args)

