clear all;
clc;

%% Name: Anubhav Joshi
% Roll No: 21101008

%% Data
E=2*10^9;    % Youngs modulus 
h=1;         % depth
v=0.25;      % poissons ratio
q=4000;      % UDL in N/m^2.
x=10;        % x coordinate
y=10;        % y coordinate 
m=3;         % Given m=n=3
n=3;  
% Taking random value of a and b which are dimensions of plate
a=randi([10,50],200,1);    
b=randi([10,50],200,1);

% Calcuating the value of deflection
def=deflection_term(a,b,E,h,v,m,n,q,x,y);

% Plot the scattered data dimensions of plate and deflection
figure(4);
scatter3(a,b,def,'filled');
xlabel('Length a');
ylabel('width b');
zlabel('Deflection','Fontsize',18);
title('Scattered Data','Fontsize',20);

%% Feature scaling
def=(def-mean(def))./(std(def));   % Feature scaling for deflection
a=(a-mean(a))./(std(a));           % Feature scaling for dimension a
b=(b-mean(b))./(std(b));           % Feature scaling for dimension b

%% Spliting the data
train_x=[a(1:150) b(1:150)];      % 2 features(dimension of plate), 150 training data 
test_x=[a(151:200) b(151:200)];   % 50 test data
train_y=def(1:150);               %  150 training data (deflection)
test_y=def(151:200);              % Test data (True value)

%% Models

% SVM model
Md_SVM=fitrsvm(train_x,train_y);
Md_SVM.ConvergenceInfo.Converged;
... cross vaidation (KFold using 5 folds)
Md_svm_CV=fitrsvm(train_x,train_y,'Standardize',true,'KFold',10);  

% GP model
Md_gp=fitrgp(train_x,train_y);

% Neural Network Model
Md_NN=fitrnet(train_x,train_y,'Standardize',true);

%% Prediction and RMSE Calculation

%SVM 
ypred_SVM=predict(Md_SVM,test_x);
RSS_SVM=sum((test_y-ypred_SVM).^2);
RMSE_SVM=sqrt(RSS_SVM/length(ypred_SVM));
R2_SVM=1-(sum((test_y-ypred_SVM).^2))/(sum((test_y-mean(test_y)).^2))

% GP
ypred_gp=predict(Md_gp,test_x);
RSS_gp=sum((test_y-ypred_gp).^2);
RMSE_gp=sqrt(RSS_gp/length(ypred_gp));
R2_gp=1-(sum((test_y-ypred_gp).^2))/(sum((test_y-mean(test_y)).^2))

% NN 
ypred_NN=predict(Md_NN,test_x);
RSS_NN=sum((test_y-ypred_NN).^2);
RMSE_NN=sqrt(RSS_NN/length(ypred_NN));
R2_NN=1-(sum((test_y-ypred_NN).^2))/(sum((test_y-mean(test_y)).^2))

%% Plot

% Plot predicted value by SVM model
figure(1);
plot(test_y,ypred_SVM,'o');
hold on;
plot(test_y,test_y);
xlabel('True Deflection','Fontsize',18);
ylabel('Predict Deflection','Fontsize',18);
legend('Predicted value','True value');
title('SVM','Fontsize',18);

% Plot predicted value by Gaussian process Regression
figure(2);
plot(test_y,ypred_gp,'+');
hold on;
plot(test_y,test_y);
xlabel('True Deflection','Fontsize',18);
ylabel('Predict Deflection','Fontsize',18);
legend('Predicted value','True value');
title('Gaussian Process Regression','Fontsize',18);

% Plot predicted value by Neural Network Regression
figure(3);
plot(test_y,ypred_NN,'p');
hold on;
plot(test_y,test_y);
xlabel('True Deflection','Fontsize',18);
ylabel('Predict Deflection','Fontsize',18);
legend('Predicted value','True value');
title('Neural Network Regression','Fontsize',18);