clear all;
%------------------Load file and true answers -------------------------

takehome_1 = load('rdani-classified-takehome1.txt');
count_rows = size(takehome_1,1);
answer = [3 1 2 3 2 1]; 
count_vector = size(answer,2);
true_value = (repmat(answer,1,count_rows/count_vector))';                   %repeating the pattern 15000/6 times

%---------------Create confusion matrix and verify -----------------

confusion_matrix = confusionmat(true_value, takehome_1);
correct = trace(confusion_matrix);
p_error_1 = (15000 - correct) / 15000;

%----------------------------- PCA ----------------------------------

input = load('train_sp2017_v19'); 
test = load('test_sp2017_v19');
[d,N] = size(input');
k = 2;                                                                      % reduced dimension
[e_vec, s, e_val] = svd(cov(input - repmat(mean(input), 15000, 1)));
e_vec_reduced = e_vec(:, 1:k);
train_reduced = (e_vec_reduced') * input';
test_reduced = (e_vec_reduced') * test';
[d1,N1] = size(train_reduced);

%-------------------------Classification-----------------------------

class_1 = train_reduced(1:d1, 1:5000);
class_2 = train_reduced(1:d1, 5001:10000);
class_3 = train_reduced(1:d1, 10001:15000);
mu_1 = mean(class_1,2);  
mu_2 = mean(class_2,2);  
mu_3 = mean(class_3,2);  
covar_1 = cov(class_1');
covar_2 = cov(class_2');
covar_3 = cov(class_3');
store = ones(N1,1);

% -----------------Discriminant function ---------------------------
for j = 1: N1
    g1x = (-0.5)*((train_reduced(1:d1,j) - mu_1).')* inv(covar_1)* ( train_reduced(1:d1,j) - mu_1) - (.5 * log(det(covar_1))) ;
    g2x = (-0.5)*((train_reduced(1:d1,j) - mu_2).')* inv(covar_2)* ( train_reduced(1:d1,j) - mu_2) - (.5 * log(det(covar_2))) ;
    g3x = (-0.5)*((train_reduced(1:d1,j) - mu_3).')* inv(covar_3)* ( train_reduced(1:d1,j) - mu_3) - (.5 * log(det(covar_3))) ;
    if(g1x > g2x && g1x > g3x)
        store(j,1) = 1;
    elseif(g2x > g1x && g2x > g3x)
        store(j,1) = 2;
    else
        store(j,1) = 3;        
    end        
end
pattern = [1 2 3];
answer_2 = (reshape(repmat(pattern,5000,1),1,5000*length(pattern)))';
confusion_matrix_2 = confusionmat(store, answer_2);
correct_2 = trace(confusion_matrix_2);
p_error_2 = (15000 - correct_2)/15000;

%---------------Classification of test data--------------------------------

store2=ones(N1,1);
% Calculating gaussian discriminant function for each test feature vector
for j=1: N
    g1_1x= (-0.5)*((test_reduced(1:d1,j) - mu_1).')* inv(covar_1)* ((test_reduced(1:d1,j) - mu_1)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar_1))) ;
    g2_1x= (-0.5)*((test_reduced(1:d1,j) - mu_2).')* inv(covar_2)* (( test_reduced(1:d1,j) - mu_2)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar_2))) ;
    g3_1x= (-0.5)*((test_reduced(1:d1,j) - mu_3).')* inv(covar_3)* (( test_reduced(1:d1,j) - mu_3)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar_3))) ;
    if(g1_1x > g2_1x && g1_1x > g3_1x)
        store2(j,1)=1;
    elseif(g2_1x > g1_1x && g2_1x > g3_1x)
          store2(j,1)=2;
    else
         store2(j,1)=3;        
    end        
end
confusion_matrix_3 = confusionmat(store2, true_value);
correct_3=trace(confusion_matrix_3);
%finding probability of error
p_error_3=(15000-correct_3)/15000;

%---------------------- Print in a text file -----------------------------

fileID = fopen('C:\Users\Rohan\Documents\MATLAB\results_PCA.txt','w+');
fprintf(fileID, '%u \r\n', store2);
fclose(fileID);

