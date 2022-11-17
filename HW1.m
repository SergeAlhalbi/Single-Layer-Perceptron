%==========================================================================
% Project 1
%==========================================================================

%% Loading

% Load the train images
training_images = loadMNISTImages('train-images.idx3-ubyte');
% train_images(:,i) is a double matrix of size 784xi(where i = 1 to 60000)
% intensity rescale to [0,1]

training_labels = loadMNISTLabels('train-labels.idx1-ubyte');
% train_labels(i) - 60000x1 vector

testing_images = loadMNISTImages('t10k-images.idx3-ubyte');
% testing_images(:,i) is a double matrix of size 784xi(where i = 1 to 10000)
% intensity rescale to [0,1]

testing_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% test_labels(i) - 10000x1 vector

%% Task 1

% Prepare experinment data
number_of_training_images_1 = 300;
number_of_testing_images_1 = 100;
[balanced_train_image_1, balanced_train_labels_1] = balance_MNIST_selection(training_images,training_labels,number_of_training_images_1);
[balanced_test_image, balanced_test_labels] = balance_MNIST_selection(testing_images,testing_labels,number_of_testing_images_1);

training_data_1 = zeros(784, number_of_training_images_1);
training_data_label_1 = zeros(number_of_training_images_1,1);
testing_data_1 = zeros(784, number_of_testing_images_1);
testing_data_label_1 = zeros(number_of_testing_images_1,1);

for i = 1: number_of_training_images_1
    % bipolar image
    training_data_1(:,i) = (double(imbinarize(balanced_train_image_1(:,i),0.5))-0.5)*2;
    training_data_label_1(i) = balanced_train_labels_1(i);
end

for i = 1: number_of_testing_images_1
    testing_data_1(:,i) = (double(imbinarize(balanced_test_image(:,i),0.5))-0.5)*2;
    testing_data_label_1(i) = balanced_test_labels(i);
end

%-------------------------------------
% Training with the first learning rate
%-------------------------------------

Learning_rate_1 = 0.01;

[Error_for_plot_1_a, W_1_a, Total_iterations_1_a] = SingleLayerPerceptron(training_data_1,training_data_label_1,784,10,Learning_rate_1,0.001,"off");

figure(1);
plot(Error_for_plot_1_a);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_1,2),Learning_rate_1);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_1,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_1(:,m_test),[1,10]);

    Net_test = sum( W_1_a .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_1(m_test)+1) = 1;
    
    if y_test(testing_data_label_1(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_1(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(2);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_1,2),size(testing_data_1,2),Learning_rate_1);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_1,2),size(testing_data_1,2),Learning_rate_1);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%-------------------------------------
% Training with the second learning rate
%-------------------------------------

Learning_rate_2 = 0.05;

[Error_for_plot_1_b, W_1_b,Total_iterations_1_b] = SingleLayerPerceptron(training_data_1,training_data_label_1,784,10,Learning_rate_2,0.001,"off");

figure(3);
plot(Error_for_plot_1_b);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_1,2),Learning_rate_2);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the second learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_1,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_1(:,m_test),[1,10]);

    Net_test = sum( W_1_b .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_1(m_test)+1) = 1;
    
    if y_test(testing_data_label_1(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_1(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(4);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_1,2),size(testing_data_1,2),Learning_rate_2);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_1,2),size(testing_data_1,2),Learning_rate_2);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%-------------------------------------
% Training with the third learning rate
%-------------------------------------

Learning_rate_3 = 0.1;

[Error_for_plot_1_c, W_1_c,Total_iterations_1_c] = SingleLayerPerceptron(training_data_1,training_data_label_1,784,10,Learning_rate_3,0.001,"off");

figure(5);
plot(Error_for_plot_1_c);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_1,2),Learning_rate_3);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the third learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_1,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_1(:,m_test),[1,10]);

    Net_test = sum( W_1_c .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_1(m_test)+1) = 1;
    
    if y_test(testing_data_label_1(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_1(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(6);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_1,2),size(testing_data_1,2),Learning_rate_3);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_1,2),size(testing_data_1,2),Learning_rate_3);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%% Task 2

% Prepare experinment data
number_of_training_images_2 = 1000;
number_of_testing_images_2 = 300;
[balanced_train_image_2, balanced_train_labels_2] = balance_MNIST_selection(training_images,training_labels,number_of_training_images_2);
[balanced_test_image, balanced_test_labels] = balance_MNIST_selection(testing_images,testing_labels,number_of_testing_images_2);

training_data_2 = zeros(784, number_of_training_images_2);
training_data_label_2 = zeros(number_of_training_images_2,1);
testing_data_2 = zeros(784, number_of_testing_images_2);
testing_data_label_2 = zeros(number_of_testing_images_2,1);

for i = 1: number_of_training_images_2
    % bipolar image
    training_data_2(:,i) = (double(imbinarize(balanced_train_image_2(:,i),0.5))-0.5)*2;
    training_data_label_2(i) = balanced_train_labels_2(i);
end

for i = 1: number_of_testing_images_2
    testing_data_2(:,i) = (double(imbinarize(balanced_test_image(:,i),0.5))-0.5)*2;
    testing_data_label_2(i) = balanced_test_labels(i);
end

%-------------------------------------
% Training with the first learning rate
%-------------------------------------

Learning_rate_1 = 0.01;

[Error_for_plot_2_a, W_2_a, Total_iterations_2_a] = SingleLayerPerceptron(training_data_2,training_data_label_2,784,10,Learning_rate_1,0.001,"off");

figure(7);
plot(Error_for_plot_2_a);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_2,2),Learning_rate_1);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_2,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_2(:,m_test),[1,10]);

    Net_test = sum( W_2_a .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_2(m_test)+1) = 1;
    
    if y_test(testing_data_label_2(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_2(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(8);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_2,2),size(testing_data_2,2),Learning_rate_1);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_2,2),size(testing_data_2,2),Learning_rate_1);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%-------------------------------------
% Training with the second learning rate
%-------------------------------------

Learning_rate_2 = 0.05;

[Error_for_plot_2_b, W_2_b,Total_iterations_2_b] = SingleLayerPerceptron(training_data_2,training_data_label_2,784,10,Learning_rate_2,0.001,"off");

figure(9);
plot(Error_for_plot_2_b);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_2,2),Learning_rate_2);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the second learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_2,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_2(:,m_test),[1,10]);

    Net_test = sum( W_2_b .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_2(m_test)+1) = 1;
    
    if y_test(testing_data_label_2(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_2(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(10);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_2,2),size(testing_data_2,2),Learning_rate_2);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_2,2),size(testing_data_2,2),Learning_rate_2);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%-------------------------------------
% Training with the third learning rate
%-------------------------------------

Learning_rate_3 = 0.1;

[Error_for_plot_2_c, W_2_c,Total_iterations_2_c] = SingleLayerPerceptron(training_data_2,training_data_label_2,784,10,Learning_rate_3,0.001,"off");

figure(11);
plot(Error_for_plot_2_c);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_2,2),Learning_rate_3);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the third learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_2,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_2(:,m_test),[1,10]);

    Net_test = sum( W_2_c .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_2(m_test)+1) = 1;
    
    if y_test(testing_data_label_2(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_2(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(12);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_2,2),size(testing_data_2,2),Learning_rate_3);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_2,2),size(testing_data_2,2),Learning_rate_3);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%% Task 3

% Prepare experinment data
number_of_training_images_3 = 1000;
number_of_testing_images_3 = 300;
[training_data_3, training_data_label_3] = balance_MNIST_selection(training_images,training_labels,number_of_training_images_3);
[testing_data_3, testing_data_label_3] = balance_MNIST_selection(testing_images,testing_labels,number_of_testing_images_3);

%-------------------------------------
% Training with the first learning rate
%-------------------------------------

Learning_rate_1 = 0.01;

[Error_for_plot_3_a, W_3_a, Total_iterations_3_a] = SingleLayerPerceptron(training_data_3,training_data_label_3,784,10,Learning_rate_1,0.001,"on");

figure(13);
plot(Error_for_plot_3_a);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_3,2),Learning_rate_1);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the first learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_3,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_3(:,m_test),[1,10]);

    Net_test = sum( W_3_a .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_3(m_test)+1) = 1;
    
    if y_test(testing_data_label_3(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_3(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(14);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_3,2),size(testing_data_3,2),Learning_rate_1);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_3,2),size(testing_data_3,2),Learning_rate_1);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%-------------------------------------
% Training with the second learning rate
%-------------------------------------

Learning_rate_2 = 0.05;

[Error_for_plot_3_b, W_3_b,Total_iterations_3_b] = SingleLayerPerceptron(training_data_3,training_data_label_3,784,10,Learning_rate_2,0.001,"on");

figure(15);
plot(Error_for_plot_3_b);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_3,2),Learning_rate_2);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the second learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_3,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_3(:,m_test),[1,10]);

    Net_test = sum( W_3_b .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_3(m_test)+1) = 1;
    
    if y_test(testing_data_label_3(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_3(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(16);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_3,2),size(testing_data_3,2),Learning_rate_2);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_3,2),size(testing_data_3,2),Learning_rate_2);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);

%-------------------------------------
% Training with the third learning rate
%-------------------------------------

Learning_rate_3 = 0.1;

[Error_for_plot_3_c, W_3_c,Total_iterations_3_c] = SingleLayerPerceptron(training_data_3,training_data_label_3,784,10,Learning_rate_3,0.001,"on");

figure(17);
plot(Error_for_plot_3_c);
grid;
str = sprintf('%d training data, learning rate is %g',size(training_data_3,2),Learning_rate_3);
title(str);
xlabel('iteration');
ylabel('Mean square error');

% Testing with the third learning rate
Threshold = 0;

% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);

for m_test = 1 : size(testing_data_3,2)
 % ex: from 1 to 100
    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(testing_data_3(:,m_test),[1,10]);

    Net_test = sum( W_3_c .* test_x_i);
    y_test(Net_test > Threshold) = 1;
    y_real(1,testing_data_label_3(m_test)+1) = 1;
    
    if y_test(testing_data_label_3(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(testing_data_label_3(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);
end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

x_axis = 0:1:9;
figure(18);
subplot(1,2,1);
bar(x_axis, TPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_3,2),size(testing_data_3,2),Learning_rate_3);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('True Positive Rate','FontSize',20);

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training data, %d testing data,\n learning rate is %g'...
    ,size(training_data_3,2),size(testing_data_3,2),Learning_rate_3);
title(str_test,'FontSize',10);
xlabel('digits','FontSize',20);
ylabel('False Positive Rate','FontSize',20);
%==========================================================================