function alphN = updatepair( Para , i , j , s , alph , Ei, Ej , H , idmx )

    tol = 1e-10;
    alphN = alph;
    id_case1 = logical(Para.idu2 + Para.idng);
    id_case2 = logical(Para.idu1 + Para.idps);
    Ci = Para.CCu(i);  Cj = Para.CCu(j);  
    si = s(i);                        sj = s(j); 
    alphi = alph(i);            alphj = alph(j); 
    Hii = H(i,i);      Hij = H(i,j);      Hjj = H(j,j); 
    
    if si == 1
        p = si*alphi + sj*alphj;
    else
        p = - si*alphi - sj*alphj;
    end
    eta = Hii + Hjj - 2*si*sj*Hij; 
    if eta > 0
        alphNi = alphi +  si*(Ej-Ei ) / eta;  
        if si~=sj 
            Lw = max(0, p);       Up = min(Cj+p, Ci);
        else
            Lw = max(0, p-Cj);    Up = min(p, Ci);
        end
        if alphNi>=Up
            alphNi = Up; 
        elseif alphNi<=Lw
            alphNi = Lw; 
        end
        alphNj = alphj + si*sj*(alphi - alphNi); 
        
    elseif eta <= 0 
        
        Lw = 0;      Up = Ci;
        Lw1 = alphj + si*sj*(alphi-Lw);  
        Up1 = alphj + si*sj*(alphi-Up); 
        Wlw = 0.5*Lw^2*Hii + 0.5*Lw1^2*Hjj + Lw*Lw1*Hij ...
                    + Lw * ( si*Ei - alphi*Hii - alphj*Hij ) ...
                    + Lw1*( sj*Ej - alphj*Hij - alphj*Hjj ); 
        Wup = 0.5*Up^2*Hii + 0.5*Up1^2*Hjj + Up*Up1*Hij ...
                    + Up * (si*Ei - alphi*Hii - alphj*Hij ) ...
                    + Up1*( sj*Ej - alphj*Hij - alphj*Hjj ); 
        if Wlw <= Wup 
            alphNi = Lw;     alphNj = Lw1;
        elseif Wlw > Wup 
            alphNi = Up;    alphNj = Up1;
        end
        
    end 
    alphNi(alphNi<tol) = 0;    alphNj(alphNj<tol) = 0;    
    if (idmx == 1 && s(id_case1)'*alph(id_case1)<=tol) || (idmx == 2&& s(id_case2)'*alph(id_case2)>=-tol || idmx == 3)
       alphN(i) = alphNi;                 alphN(j) = alphNj; 
    end
    
end 


