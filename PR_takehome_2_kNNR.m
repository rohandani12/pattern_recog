clear all;
%----------------Load data -------------------------

train = load('train_sp2017_v19'); 
test = load('test_sp2017_v19');

%----------------Try dist --------------------------
tic
distance = sqrt( bsxfun(@plus,sum(train.^2,2),sum(test.^2,2)') - 2*(train*test') );
toc
tic
[sorted, index] = sort(distance,1);
toc
store = zeros(15000,1);

%----------------1-NNR--------------------------
for i = 1:15000
    if (index(1,i)<5000)
        store(i,1) = 1;
    elseif (index(1,i)>=5000 && index(1,i)<10000)
        store(i,1) = 2;
    elseif(index(1,i)>=10000)
        store(i,1) = 3;
    end
end
answer = [3 1 2 3 2 1]; 
count_vector = size(answer,2);
true_value = (repmat(answer,1,15000/6))';                   %repeating the pattern 15000/6 times

%---------------Create confusion matrix and verify -----------------

confusion_matrix_1 = confusionmat(true_value, store);
correct = trace(confusion_matrix_1);
p_error_1 = (15000 - correct) / 15000;
count_1 = 0; count_2 = 0; count_3 = 0;

%---------------------- Print in a text file -----------------------------

fileID = fopen('C:\Users\Rohan\Documents\MATLAB\results_1nnr.txt','w+');
fprintf(fileID, '%u \r\n', store);
fclose(fileID);

%----------------3-NNR--------------------------
for i = 1:15000
    count_1 = 0; count_2 = 0; count_3 = 0;
    for j = 1:3
        if (index(j,i)<5000)
            count_1 = count_1 + 1;
            if(count_1 == 2)
                store(i,1) = 1;
            end
        elseif (index(j,i)>=5000 && index(j,i)<10000)
            count_2 = count_2 + 1;
            if(count_2 == 2)
                store(i,1) = 2;
            end
    elseif(index(j,i)>=10000)
         count_3 = count_3 + 1;
            if(count_3 == 2)
                store(i,1) = 3;
            end
        end
    end
end
%---------------Create confusion matrix and verify -----------------

confusion_matrix_2 = confusionmat(true_value, store);
correct = trace(confusion_matrix_2);
p_error_2 = (15000 - correct) / 15000;

%---------------------- Print in a text file -----------------------------

fileID = fopen('C:\Users\Rohan\Documents\MATLAB\results_3nnr.txt','w+');
fprintf(fileID, '%u \r\n', store);
fclose(fileID);

%----------------5-NNR--------------------------
for i = 1:15000
    count_1 = 0; count_2 = 0; count_3 = 0;
    for j = 1:5
        if (index(j,i)<5000)
            count_1 = count_1 + 1;
        elseif (index(j,i)>=5000 && index(j,i)<10000)
            count_2 = count_2 + 1;
            
        elseif(index(j,i)>=10000)
            count_3 = count_3 + 1;
        end
    end    
    if( max([count_1, count_2, count_3]) == count_1)
        store(i,1) = 1;
    elseif( max([count_1, count_2, count_3]) == count_2)
        store(i,1) = 2;
    else
        store(i,1) = 3;        
    end
end


%---------------Create confusion matrix and verify -----------------

confusion_matrix_3 = confusionmat(true_value, store);
correct = trace(confusion_matrix_3);
p_error_3 = (15000 - correct) / 15000;

%---------------------- Print in a text file -----------------------------

fileID = fopen('C:\Users\Rohan\Documents\MATLAB\results_5nnr.txt','w+');
fprintf(fileID, '%u \r\n', store);
fclose(fileID);


        
        

