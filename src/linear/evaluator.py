# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score


# class ModelEvaluator:
#     @staticmethod
#     def accuracy(x_test, y_test: np.ndarray, model):
#         y_pred = model.predict(x_test)
#         return r2_score(y_test, y_pred)
    
#     @staticmethod
#     def plot_accuracy(x_train, y_train, x_test, y_test, loss, optimizer, loss_name, ranges):
#         scores = []
#         interval = [i for i in range(10, ranges + 1, 10)]
#         for i in interval:
#             model = LinearRegression()
#             model.compile(
#                 optimizer=optimizer,
#                 loss=loss
#             )
#             model.train(
#                 x_train,
#                 y_train,
#                 epochs=i,
#                 verbose=3
#             )
#             scores.append(ModelEvaluator.accuracy(x_test, y_test, model))

#         plt.plot(interval, scores, 'o-')
#         plt.title(f"R2 Scores VS. Epochs ({loss_name} loss)")
#         plt.xlabel("Number of Epochs")
#         plt.ylabel("R2 Score")
#         plt.grid(True)
#         return interval, scores
