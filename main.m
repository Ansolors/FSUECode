clear
clc

%% --------  Data  Path -------------
DP = '.\Data\Cleveland.mat'; 
% ---------------------
% DP = '.\Data\Monks1.mat'; 
% ---------------------
% DP = '.\Data\New_thyroid.mat'; 
% ---------------------
% DP = '.\Data\SPECT.mat';
% ---------------------
% DP = '.\Data\WOBC.mat'; 
% ---------------------
% DP = '.\Data\PimaIndian.mat';

%% --------  Loading Data  -------------
load(DP);  

%% --------- Predict ----------
[ PredictY , model ] = FSUE( Data.TstX , Data , Para );
TestAc = sum(PredictY == Data.TstY)/length(PredictY)*100
