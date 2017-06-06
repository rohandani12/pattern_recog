clear all;

%------------------------Load data--------------------------------

train = load('train_sp2017_v19');
test = load('test_sp2017_v19');

%---------------------separate classes ---------------------------

c1 = train(1:5000, :);
c2 = train(5001:10000, :);
c3 = train(10001:15000, :);

eta = 0.3;kmax = 1000;k = 1;
n1 = size(c1,1);
n2 = size(c2,1);
n3 = size(c3,1);

%---------------------Separate class 1-----------------------------

n = n1+n2+n3;
Y = [ones(n1,1) c1 ; -ones(n2,1) -c2; -ones(n3,1) -c3];
b = [(n/n1)*ones(n1,1) ; (n/(n2+n3))*ones(n2+n3,1)];
pinvY = pinv(Y);
a1 = pinvY*b;
while (k < kmax)
  k = k+1;
  e = Y*a1-b;
  eplus = (e+abs(e));
  b = b+eta*eplus;
  a1 = pinvY*b;
end

%----------------------Separate class 2  --------------------------

Y = [ones(n2,1) c2 ; -ones(n3,1) -c3; -ones(n1,1) -c1];
b = [(n/n2)*ones(n2,1) ; (n/(n3+n1))*ones(n3+n1,1)];
pinvY = pinv(Y);
a2 = pinvY*b;
k = 1;
while (k < kmax)
  k = k+1;
  e = Y*a2-b;
  eplus = (e+abs(e));
  b = b+eta*eplus;
  a2 = pinvY*b;
end

%----------------------Separating class 3 --------------------------

Y = [ones(n3,1) c3 ; -ones(n1,1) -c1; -ones(n2,1) -c2];
b = [(n/n3)*ones(n3,1) ; (n/(n1+n2))*ones(n1+n2,1)];
pinvY = pinv(Y);
a3 = pinvY*b;
k = 1;
while (k < kmax)
  k = k+1;
  e = Y*a3-b;
  eplus = (e+abs(e));
  b = b+eta*eplus;
  a3 = pinvY*b;
end
store = ones(15000,1);
test = [ones(15000,1) test];
cl1 = test*a1;
cl3 = test*a3;
cl2 = test*a2;
for j=1:15000
    if(max([cl1(j,1) cl2(j,1) cl3(j,1)]) == cl1(j,1))
        store(j,1)=1;
    elseif(max([cl1(j,1) cl2(j,1) cl3(j,1)]) == cl3(j,1))
        store(j,1)=3;
    else
        store(j,1)=2;
    end
end

answer = [3 1 2 3 2 1]; 
count_vector = size(answer,2);
true_value = (repmat(answer,1,15000/6))';                   

%---------------Create confusion matrix and verify -----------------

confusion_matrix = confusionmat(true_value, store);
correct = trace(confusion_matrix);
p_error = (15000 - correct) / 15000;

%---------------------- Print in a text file -----------------------------

fileID = fopen('C:\Users\Rohan\Documents\MATLAB\results_hokashyap.txt','w+');
fprintf(fileID, '%u \r\n', store);
fclose(fileID);


