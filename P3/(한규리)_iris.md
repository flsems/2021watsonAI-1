# 속성 3개
# 중간층
from sklearn.datasets import load_iris
iris=load_iris()
X=tf.constant(iris.data[:,[0,1,2]],dtype=tf.float32)
->설명1: iris.csv의 데이터의 0,1,2번째 열의 값인 SepalLengthCm, SepalWidthCm, PetalLengthCm를 x값으로 지정, 상수의 데이터형은 tf.float32
y=tf.constant(iris.data[:,3],dtype=tf.float32)

w=tf.Variable(tf.random.normal([3,5]))
b=tf.Variable(tf.random.normal([5]))

u=tf.nn.relu(X@w+b)
->설명2: 히든 레이어 'Relu'함수를 이용해 신경망에 가중치 W과 편향 b를 적용해서 텐서플로우에서 기본적으로 제공하는 활성화 함수인 ReLU 함수를 적용
# 150x5
ww=tf.Variable(tf.random.normal([5,5]))
bb=tf.Variable(tf.random.normal([5]))

uu=tf.nn.relu(u@ww+bb)

www=tf.Variable(tf.random.normal([5,1]))
bbb=tf.Variable(tf.random.normal([]))

pred_y=uu@www+bbb
->설명3: 출력층을 만들기 위해 두번째 가중치 w와 편향 b를 적용하여 최종 모델을 만듦

mse=tf.reduce_mean(tf.square(y-pred_y))
->설명4: mse : 평균제곱근 오차, 실제값 y에서 예측 값 pred_y의 차의 제곱근의 평균 값

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op=optimizer.minimize(mse)
->설명5: 오차가 최소화(=최적화)가 되기 위해 학습률 learning_rate=0.001로 설정,
optimizer : 경사하강법, minimize의 대상이 되는 mse는 평균 제곱근의 오차이다.
GradientDescentOptimizer 함수는 경사하강법을 구현한 함수임
경사는 평균제곱근의 오차를 가중치로 미분한 값, minimize 함수는 최소화한 결과를 반환함

costs=[]

tf.global_variables_initializer().run()

for i in range(300):
    sess.run(train_op)
    costs.append(mse.eval())
plt.plot(costs)
