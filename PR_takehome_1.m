clc;
clear all;
close all;

%------------------------- Loading data ------------------------------

data=load('train_sp2017_v19'); 
test_data=load('test_sp2017_v19');

%----------------------- Transposing data ----------------------------

test_data=(test_data.');
data_transposed=(data');

%------------ Initializing data into sets for data entry ----------------

dataset_1=ones(5000,4);
dataset_2=ones(5000,4);
dataset_3=ones(5000,4);

%------------------ Entering data into the sets -----------------------

dataset_1(1:5000,:)=data(1:5000,:);
dataset_2(1:5000,:)=data(5001:10000,:);
dataset_3(1:5000,:)=data(10001:15000,:);   


%------------------ Mean and covariance of sets -----------------------

mu1=mean((dataset_1)).'; 
mu2=mean((dataset_2)).';  
mu3=mean((dataset_3)).';  
covariance_1=cov(dataset_1);
covariance_2=cov(dataset_2);
covariance_3=cov(dataset_3);

%-------------Initializing Gaussian discriminant function -----------------

g1x = ones(15000,1);
g2x = ones(15000,1);
g3x = ones(15000,1);
store = ones(15000,1);

%---------------------- Calculating Gaussian DF --------------------------

g1x= (-0.5)*((data_transposed - mu1).')* (inv(covariance_1))* (( data_transposed - mu1)) - (2 * log(2*pi)) - (.5 * log(det(covariance_1))) ;
g2x= (-0.5)*((data_transposed - mu1).')* (inv(covariance_2))* (( data_transposed - mu2)) - (2 * log(2*pi)) - (.5 * log(det(covariance_2))) ;
g3x= (-0.5)*((data_transposed - mu1).')* (inv(covariance_3))* (( data_transposed - mu3)) - (2 * log(2*pi)) - (.5 * log(det(covariance_3))) ;
for j=1:15000
    if(g1x > g2x && g1x > g3x)
        store(j,1)=1;
    elseif(g2x > g1x && g2x > g3x)
          store(j,1)=2;
    else
         store(j,1)=3;
    end
end
err2=0;err3=0;err1=0;
for i=1:15000
    if (i <= 5000) && ((store(i,1) == 2) || (store(i,1) == 3))
            err1 = err1+1;
    elseif ((i > 5000) && (i<=10000)) && ((store(i,1) == 1) || (store(i,1) == 3))
            err2 = err2 + 1;
    elseif ((i > 10000) &&(i <= 15000)) && ((store(i,1) == 2) || (store(i,1) == 1))
            err3 = err3+1;
    end
end
err=(err1+err2+err3)/15000;