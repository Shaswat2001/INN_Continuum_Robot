data = xlsread('../MLDataset_quat.csv');
X = data(2:end,2:8);
Y = data(2:end,9:end);

values = randi([1,size(X,1)],[20000 1]);
X = X(values',:);
Y = Y(values',:);
cv = cvpartition(size(X,1),'HoldOut',0.15);
idx = cv.test;

X_train=X(~idx,:);
X_test=X(idx,:);
Y_train=Y(~idx,:);
Y_test=Y(idx,:);

L2_train=Y_train(:,2);
L2_test=Y_test(:,2);

L1_train=Y_train(:,1);
L1_test=Y_test(:,1);

L3_train=Y_train(:,3);
L3_test=Y_test(:,3);

L4_train=Y_train(:,4);
L4_test=Y_test(:,4);