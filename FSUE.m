function [ PredictY , model ] = FSUE( ValX , Trn , Para )
% Solving [Feature Selection by Universum Embedding SVC] via SMO and DCD.  
% PRIMAL: 
% _______________________________ Input  _______________________________
%      Trn.X  -  m x n matrix, explanatory variables in training data 
%      Trn.Y  -  m x 1 vector, response variables in training data 
%      Trn.Ux  -  (4*n) x n vector, Universum data
%      Trn.Uy  -  (4*n) x 1 matrix, Universum label 
%      ValX   -  mt x n matrix, explanatory variables in Validation data 
%      Para.p1  -  the emperical risk parameter C1 
%      Para.p2  -  the Universum parameter C2 
%      Para.p3  -  the Universum parameter Cu 
%      Para.p4  -  the parameter epsilon 
% ______________________________ Output  ______________________________
%     PredictY  -  mt x 1 vector, predicted response variables for TestX 
%     model  -  model related info.
% 
% Written by Chunna Li.
% Modefied by Lingwei Huang, lateset update: 2023.11.1. 
% Copyright 2021  Chunna Li & Lingwei Huang. 

%% Input     
    Xneg = Trn.X(Trn.Y==-1,:); Xpos = Trn.X(Trn.Y==1,:);
    Yneg = Trn.Y(Trn.Y==-1,:); Ypos = Trn.Y(Trn.Y==1,:);
    Trn.X = [Xneg; Xpos];  Trn.Y = [Yneg; Ypos]; 
    clear Xneg Xpos Yneg Ypos
    
    X = Trn.X;      	   Y = Trn.Y;        % ori data
    Ux = Trn.Ux;        Uy = Trn.Uy;   % uni data 
    XU = [ X ; Ux ];    YU = [ Y ; Uy ]; % expended data
    C = Para.p1;      
    Cu = Para.p2;
    Cr = Para.p3;      
    epsilon = Para.p4; 
    tol = 1e-10;      eps3 = 10;    wei = 0.6;  wei2 = 0.95; 
    itmax = 5;        itSMO = 4;      itDCD = 5;  
    
%% Initilization     
    [m,n] = size(X);                 
    ps = sum(Y==1);               ng = sum(Y==-1); 

    Sn = length(Uy)*0.25;
    uni = Ux(1:Sn, :);  
    
    em = ones(m,1);   eSn = ones(Sn,1);  e2Sn = ones(2*Sn,1); em2Sn = ones(m+2*Sn,1);       
    zm4Sn = zeros(m+4*Sn,1);  z2Sn = zeros(2*Sn,1);   
        
    CCu = [ C*em ; Cu*e2Sn ; Cr*e2Sn ]; 
    idng = YU==-1;    idps = YU==1;   idng(m+1:end) = 0;  idps(m+1:end) = 0;   
    idu = false(m+4*Sn,1); idu1 = idu;  idu2 = idu; 
    idu1(m+1:m+Sn) =1;     
    idu2(m+Sn+1:m+2*Sn) =1;  
    idclass = logical(idng + idps + idu1 + idu2);
    EY = sparse(diag(YU));  

    rand1 = ones(ng,1);  
    rand2 = ones(ps,1); 
    rand3 = ones(Sn,1);    
    rand4 = ones(Sn,1);    
      
    alph = [ (sum(rand4))*rand1/(sum(rand1)) ; (sum(rand3))*rand2/(sum(rand2))  ; rand3 ; rand4 ; z2Sn ];
    model.alpha0 = alph;
    alph = alph * C/max(alph);
    
    if isfield(Para, 'alpha0')
        alph = Para.alpha0 * C/max(Para.alpha0);
    end
    
    Para.CCu = CCu;  Para.m = m;  Para. n = n; 
    Para.idu1 = idu1; Para.idu2 = idu2; Para.idng = idng; Para.idps = idps; 
    
%% ==== On expended data (Alogrithm 1)====   

	K = XU * XU'; K = (K+K')/2;     K=full(K);
    H = EY * K * EY;   
    g = [ -epsilon*em2Sn ; z2Sn ];
    s = [ Y ; -eSn ; +eSn ;  -eSn ; +eSn ];  
    itt = 0;
    
% %         ----------- begin finding indices ----------
    id_0 = alph<tol;                 alph(id_0) = 0;
    id_C = CCu-alph<tol;    alph(id_C) = CCu(id_C); 
    id_0C = 0<alph  &  alph<CCu;

    [ idng_0, idng_0C, idng_C, idu1_0, idu1_0C, idu1_C, ...
      idps_0,  idps_0C,  idps_C, idu2_0, idu2_0C, idu2_C ] ...
         = deal( false(m+4*Sn,1) );
    idng_0(idng) = id_0(idng);  % neg 
    idng_C(idng) = id_C(idng); 
    idng_0C(idng) = id_0C(idng); 
    idps_0(idps) = id_0(idps);   % pos 
    idps_C(idps) = id_C(idps);
    idps_0C(idps) = id_0C(idps);
    idu1_0(idu1) = id_0(idu1);   % u1 
    idu1_C(idu1) = id_C(idu1);
    idu1_0C(idu1) = id_0C(idu1);
    idu2_0(idu2) = id_0(idu2);   % u2 
    idu2_C(idu2) = id_C(idu2);
    idu2_0C(idu2) = id_0C(idu2);

    low_a = logical( idng_0 + idng_0C + idu2_0C + idu2_C ); 
    up_a = logical( idng_0C + idng_C + idu2_0 + idu2_0C ); 
    low_b = logical( idps_0C + idps_C + idu1_0 + idu1_0C ); 
    up_b = logical( idps_0 + idps_0C + idu1_0C + idu1_C ); 

    Halph  = H * alph;
    nbW = Halph + g; 
    snbW = s .* nbW;  
    % %         ----------- end finding indices ----------
    for it1 = 1 : itmax
        for it2 = 1 : itSMO
            alphN = alph; 
            itt = itt + 1;
            VP = bsxfun( @minus , snbW , snbW' );    % low-up>0 violating 
            VP_active = VP(logical(low_a+low_b),logical(up_a+up_b)); 
        
            mk = maxk(VP_active,1);    
            lmk = length(mk);       if lmk==0, break, end
            mxVP = mk(randperm(lmk,1)); 
            [vpi,vpj] = find(VP==mxVP);   
            vpi=vpi(1); vpj=vpj(1); 
            if low_a(vpi)+up_a(vpj) == 2            
                idmx = 1;       
            elseif low_b(vpi)+up_b(vpj) == 2            
                idmx = 2;       
            else
                idmx = 3;
            end
            Evpi = s(vpi) * Halph(vpi) - s(vpi) * epsilon;
            Evpj = s(vpi) * Halph(vpj) - s(vpj) * epsilon;
            alph =  wei*updatepair(Para,vpi,vpj,s,alph,Evpi,Evpj,H, idmx) + (1-wei)*alph;
        end
% % %         ----- DCD ----
        for it3 = 1: itDCD
            itt = itt + 1;
            nbWP = zm4Sn; 
            alphDCD = alph;
            for i = m+2*Sn+1: m+4*Sn % alg(2)
                nbW(i) = H(i,:)* alph + g(i);
                if alph(i) == 0
                    nbWP(i) = min(0, nbW(i));
                elseif alph(i) == Cr
                    nbWP(i) = max(0, nbW(i));
                else
                    nbWP(i) = nbW(i);
                end
                if nbWP(i) ~= 0 
                   alph(i) = min( max( alph(i)-nbW(i)/H(i,i), 0), Cr ); 
                end
            end
            alph = wei2*alph + (1-wei2)*alphDCD;    
        end
    end
% % %     ----final alpha and b & prediction -----
    
    wx = K * EY * alph; % w*x 
    [Flow_a,Fup_a,Flow_b,Fup_b] = deal( zm4Sn );
    
    Flow_a(idng_0|idng_0C) =  wx(idng_0|idng_0C) + epsilon;
    Flow_a(idu2_0C|idu2_C) =  wx(idu2_0C|idu2_C) - epsilon;
    Fup_a(idng_0C|idng_C) =  wx(idng_0C|idng_C) + epsilon;
    Fup_a(idu2_0|idu2_0C) =  wx(idu2_0|idu2_0C) - epsilon;
    
    Flow_b(idps_0C|idps_C) =  wx(idps_0C|idps_C) - epsilon;
    Flow_b(idu1_0|idu1_0C) =  wx(idu1_0|idu1_0C) + epsilon;
    Fup_b(idps_0|idps_0C) =  wx(idps_0|idps_0C) - epsilon;
    Fup_b(idu1_0C|idu1_C) =  wx(idu1_0C|idu1_C) + epsilon;
    
    bup_a = min(Fup_a);       blow_a = max(Flow_a);
    bup_b = min(Fup_b);      blow_b = max(Flow_b);
    
    Bup_a = min(bup_a,bup_b);
    Blow_b = max(blow_a,blow_b); 
    b1 = 0.5*(blow_a+Bup_a);
    b2 = 0.5*(Blow_b+bup_b);
    b = 0.5*(b1+b2);
    
    id_0 = alph<tol;                 alph(id_0) = 0;
    id_C = CCu-alph<tol;    alph(id_C) = CCu(id_C); 
    id_0C = 0<alph  &  alph<CCu;
    
%% Prediction & Output   
    wxb =  ValX * XU'* EY * alph - b;
    PredictY = sign(wxb);

    idw0 = (id_0(m+2*Sn+1:m+3*Sn)).*(id_0(m+3*Sn+1:end));
    idw0 = ~logical(idw0);
    R = (nnz(idw0) + n - Sn) / n;  
    
    if R == 1
        idw0 = FSidRevise( idw0, uni*XU'*EY*alph, eps3 ); 
    end

    w = XU' * EY * alph;        
    w(idw0) = 0;
    model.w = w; 
    model.w_ind = ~idw0; 
    
    function idw0 = FSidRevise( idw0 , wu , eps3 ) 
    n = length(wu);
    Abs =  abs(wu);
    Desc = sort(Abs,'descend'); 
    Thrld = [];
    Dif = - diff(Desc) + 1e-8;  
    
    for u = 1:n-2
        DifR = Dif(u)  /  Dif(u+1);
        if DifR >= eps3
            Thrld = Desc(u+1);  break
        elseif 1/DifR >= eps3
            Thrld = Desc(u+2);  break
        end
    end
    
    if  ~isempty(Thrld)                      
        idw0( Abs > Thrld ) = 0;      
    end
    
    if nnz(idw0) == n 
        idw0( Abs == max(Abs) ) = 0; 
    end
    
    end % end func 


    
end


