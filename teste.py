import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#df = pd.read_csv('train.csv')
#df.head(10)

# Dados Dummies - forjados
experiencia = np.random.normal(5, 2.8, 30)
experiencia = (experiencia ** 2) ** 0.5
X = pd.DataFrame(experiencia).sort_values(by=0, ascending=True)

salario = np.random.normal(50000, 20000,30)
salario = (salario ** 2) ** 0.5
y = pd.DataFrame(salario).sort_values(by=0, ascending=True)

''' Plotar x e y
plt.scatter(X, y)
plt.xlabel('Experiência')
plt.ylabel('Salário')
plt.title('Experiência x Salário')
plt.show()
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

#score
score_train = regression.score(X_train, y_train)
score_test = regression.score(X_test, y_test)

print('Score Train:', score_train)
print('Score Test:', score_test)

plt.scatter(X_train, y_train, color='green')
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regression.predict(X_train), color='red')
plt.xlabel('Experiência')
plt.ylabel('Salário')
plt.title('Tempo de Experiência x Salário')
plt.show()



