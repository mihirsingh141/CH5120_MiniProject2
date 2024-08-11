clc;
clear;
close all;

%Initial Values

A1 = 28; %(cm^2)
A2 = 32;
A3 = 28;
A4 = 32;

a1 = 0.071; a3 = 0.071; %(cm^2)
a2 = 0.057; a4 = 0.057;

kc = 1; %(V/cm)
g = 981; %(cm/s^2)

gamma1 = 0.7; gamma2 = 0.6; %constants that are determined from valve postion

k1 = 3.33; k2 = 3.35; %(cm^3/Vs)

v1 = 3;v2 = 3; %(V)

h0 = [12.4; 12.7; 1.8; 1.4];


%Linearized State Represenation Matrices

T1 = (A1/a1)*(2*h0(1)/g)^0.5;
T2 = (A2/a2)*(2*h0(2)/g)^0.5;
T3 = (A3/a3)*(2*h0(3)/g)^0.5;
T4 = (A4/a4)*(2*h0(4)/g)^0.5;

%Continuous State Space representation 
Ac = [-1/T1, 0, A3/(A1*T3), 0; 0, -1/T2, 0, A4/(A2*T4); 0, 0, -1/T3, 0; 0, 0, 0, -1/T4];
Bc = [gamma1*k1/A1, 0; 0, gamma2*k2/A2; 0, (1-gamma2)*k2/A3; (1-gamma1)*k1/A4, 0];
Cc = [kc, 0, 0, 0; 0, kc, 0, 0];

%Discrete State Space representation
Ad = expm(Ac);
Bd = integral(@(tau) expm(Ac*tau),0,0.04,'ArrayValued',true)*Bc;
Cd = Cc;



%Constraints

DUmin = 5*[-1 -1]';
DUmax = 5*[1 1]';
Umin = 0*[-1 -1]';
Umax = 20*[1 1]';


Q = 0.000001*eye(4);     %covariance matrix of noise in state equation
R = 0.000001*eye(2);     %covariance matrix of noise in measurement equation


%% Part a. 
%Assuming all states are measured, and h1 and h2 are the controlled
%variables

x = [h0(1);h0(2);h0(3);h0(4)]; %initial true state
x0 = zeros(4,1); %state before the first instant when measurements are taken

y1 = h0(1);
y2 = h0(2);

u = [0;0]; %We assume that the system starts from 'rest'

h1 = [];
h2 = [];

delu1 = [];
delu2 = [];

N = 1000; %Simulation Time
Np = 10; %Prediction Horizon
Nc = 3; %Control Horizon

q = 2; %number of outputs
n = 4; %number of state variables
m = 2; %number of inputs

Q = 0.000001*eye(4);     %covariance matrix of noise in state equation
R = 0.000001*eye(2);     %covariance matrix of noise in measurement equation



Cm = [1,0,0,0;0,1,0,0]; %Matrix used to capture the h1 and h2 variables
Cd = [1,0,0,0;0,1,0,0]; %Matrix used to capture the h1 and h2 measurements


for i=1:N
    [delU,F,phi,Ru,A,B] = controlAction(Np,Nc,n,q,m,[(x-x0);y1;y2],[13.4;13.7],Ad,Bd,Cm);

    h1 = [h1,y1];
    h2 = [h2,y2];

    if i == 1
        Kmpc = (phi'*phi + Ru)\phi'*F;
        disp('The eigen values of the closed loop system when the first move was implemented are') 
        disp(eig(A-B*Kmpc(1:2,:)));
    end

    if i == N
        Kmpc = (phi'*phi + Ru)\phi'*F;
        disp('The eigen values of the closed loop system when it stablizes close to its set point is') 
        disp(eig(A-B*Kmpc(1:2,:)));
    end

    u = u+delU(1:2);

    v = mvnrnd([0,0],R,1)'; %Measurement Noise
    y = Cd*x + v;
    
    w = mvnrnd([0,0,0,0],Q,1)'; %State Noise

    x0 = x;
    x = Ad*x + Bd*u + w;
    
    y1 = y(1);
    y2 = y(2);

    

    delu1 = [delu1;delU(1)];
    delu2 = [delu2;delU(2)];

   
end

k = 0:0.1:100;
k = k(1:end-1);

figure;
plot(k,h1,k,h2)
legend('h1','h2')
title('Outputs')



figure;
stairs(k,delu1,'b')
hold on
stairs(k,delu2,'r')
title('Zero Order Graph of Incremental Inputs')
xlabel('Time');
ylabel('Signal Value');
legend('delU1','delU2')
grid on;


%% Part b. 
%Assuming only h1 and h2 are measured but h3 and h4 are the controlled
%variables


x0 = zeros(4,1); %Initial State estimate

x_true = h0; %True initial State

y1 = h0(1); %h1 measurement
y2 = h0(2); %h2 measurement

u = [0;0]; %We assume that the system starts from 'rest'

h3 = [];
h4 = [];

delu1 = [];
delu2 = [];


Cm = [0,0,1,0;0,0,0,1]; %Matrix used to extract h3 and h4 states
Cd = [1,0,0,0;0,1,0,0]; %Matrix used to extract h1 and h2 measurements

Np = 10; %Prediction Horizon
Nc = 3; %Control Horizon

N = 10000; %Simulation Time

P0 = 0.00001*eye(4); %arbitrary initial state error covariance matrix


Q = 0.001*eye(4);     %covariance matrix of noise in state equation
R = 0.001*eye(2);     %covariance matrix of noise in measurement equation


for i=1:N
    [xm_hat,P] = kalmanFilter(Ad,Bd,Cd,Q,R,[y1;y2],x0,u,P0); %Using Kalman Filter to make state estimation
    P0 = P;

    y3 = xm_hat(3); %h3 value
    y4 = xm_hat(4); %h4 value

    %Implementing constrained MPC
    [delU,F,phi] = controlActionConstrained(Np,Nc,n,q,m,[(xm_hat-x0);y3;y4],[2.8;2.4],Ad,Bd,Cm,u,DUmin,DUmax,Umin,Umax);

    
    u = u+delU(1:2);

    w = mvnrnd([0,0,0,0],Q,1)'; %State Noise
    x_true = Ad*x_true + Bd*u + w;
    
    v = mvnrnd([0,0],R,1)'; %Measurement Noise
    y = Cd*x_true + v;

    y1 = y(1); %h1 measurement
    y2 = y(2); %h2 measurement
    
    x0 = xm_hat;
    

    h3 = [h3,y3];
    h4 = [h4;y4];

    delu1 = [delu1;delU(1)];
    delu2 = [delu2;delU(2)];

    

end



k=0:0.1:1000;
k = k(1:end-1);

figure;
plot(k,h3,k,h4)
legend('h3','h4')
title('Outputs')

figure;
stairs(k,delu1,'b')
hold on
stairs(k,delu2,'r')
title('Zero Order Graph of Incremental Inputs')
xlabel('Time');
ylabel('Signal Value');
legend('delU1','delU2')
grid on;

%% Part c. a.
%Assuming h1 and h4 are measured, and h2 and h3 are the controlled
%variables

P0 = 0.00001*eye(4); %arbitrary initial state error covariance matrix


Q = 0.000001*eye(4);     %covariance matrix of noise in state equation
R = 0.000001*eye(2);     %covariance matrix of noise in measurement equation



x0 = zeros(4,1); %initial estimate of the state

x_true = h0; %initial true state

y1 = h0(1);
y4 = h0(4);

u = [0;0]; %We assume that systems starts from a state of 'rest'

h2 = [];
h3 = [];

delu1 = [];
delu2 = [];


Cm = [0,1,0,0;0,0,1,0]; %Matrix used to extract the h2 and h3 states
Cd = [1,0,0,0;0,0,0,1]; %Matrix used to extract the h1 and h4 measurements 

Np = 10; %Prediction horizon
Nc = 3; %Control horizon

N = 1000;

for i=1:N
    [xm_hat,P] = kalmanFilter(Ad,Bd,Cd,Q,R,[y1;y4],x0,u,P0);
    P0 = P;
    y2 = xm_hat(2);
    y3 = xm_hat(3);

    [delU,F,phi] = controlActionConstrained(Np,Nc,n,q,m,[(xm_hat-x0);y2;y3],[13.7;2.8],Ad,Bd,Cm,u,DUmin,DUmax,Umin,Umax);

    u = u+delU(1:2);

    w = mvnrnd([0,0,0,0],Q,1)'; %State Noise
    x_true = Ad*x_true + Bd*u + w;


    v = mvnrnd([0,0],R,1)'; %Measurement Noise
    y = Cd*x_true + v;
    y1 = y(1);
    y4 = y(2);
    
    x0 = xm_hat;
    

    h2 = [h2,y2];
    h3 = [h3;y3];

    delu1 = [delu1;delU(1)];
    delu2 = [delu2;delU(2)];

    

end



k=0:0.1:100;
k=k(1:end-1);

figure;
plot(k,h2,k,h3)
legend('h2','h3')
title('Outputs')
xlabel('Time')
ylabel('Signal Value')

figure;
stairs(k,delu1,'b')
hold on
stairs(k,delu2,'r')
title('Zero Order Graph of Incremental Inputs')
xlabel('Time');
ylabel('Signal Value');
legend('delU1','delU2')
grid on;

%% Part c. b. 
%Assuming h2 and h4 are measured, and h1 and h3 are the controlled
%variables



P0 = 0.00001*eye(4); %arbitrary initial state error covariance matrix


Q = 0.000001*eye(4);     %covariance matrix of noise in state equation
R = 0.000001*eye(2);     %covariance matrix of noise in measurement equation


x0 = ones(4,1); %inital estimate of state

x_true = h0; %System's true state

y2 = h0(2);
y4 = h0(4);


u = [0;0];

h1 = [];
h3 = [];

delu1 = [];
delu2 = [];


Cm = [1,0,0,0;0,0,1,0]; %Matrix used to extract the h1 and h3 states
Cd = [0,1,0,0;0,0,0,1]; %Matrix used to extract the h2 and h4 measurements

Np = 10; %Prediction Horizon
Nc = 3; %Control Horizon

N = 10000;

for i=1:N
    [xm_hat,P] = kalmanFilter(Ad,Bd,Cd,Q,R,[y2;y4],x0,u,P0);
    P0 = P;
    y1 = xm_hat(1);
    y3 = xm_hat(3);

    [delU,F,phi] = controlActionConstrained(Np,Nc,n,q,m,[(xm_hat-x0);y2;y3],[13.7;2.4],Ad,Bd,Cm,u,DUmin,DUmax,Umin,Umax);

    u = u+delU(1:2);

    w = mvnrnd([0,0,0,0],Q,1)'; %State Noise
    x_true = Ad*x_true + Bd*u + w;


    v = mvnrnd([0,0],R,1)'; %Measurement Noise
    y = Cd*x_true + v;
    y2 = y(1);
    y4 = y(2);
    
    x0 = xm_hat;
    

    h1 = [h1;y1];
    h3 = [h3;y3];

    delu1 = [delu1;delU(1)];
    delu2 = [delu2;delU(2)];

    

end



k=0:0.1:1000;
k=k(1:end-1);

figure;
plot(k,h1,k,h3)
legend('h1','h3')
title('Outputs')
xlabel('Signal Value')
ylabel('Time')

figure;
stairs(k,delu1,'b')
hold on
stairs(k,delu2,'r')
title('Zero Order Graph of Incremental Inputs')
xlabel('Time');
ylabel('Signal Value');
legend('delU1','delU2')
grid on;

function [x_hat,P] = kalmanFilter(Ad,Bd,Cd,Q,R,y,x,u,P_post)
    %Predict Step
    P_prior = Ad*P_post*Ad'+Q;
    x_prior = Ad*x + Bd*u;
    
    %Update Step
    K = P_prior*Cd'*inv(Cd*P_prior*Cd' + R);
    x_hat = x_prior + K*(y-Cd*x_prior);
    P = (eye(4)-K*Cd)*P_prior;
end



function [delU,F,phi] = controlActionConstrained(Np,Nc,n,q,m,x0,Rs0,Am,Bm,Cm,u,DUmin,DUmax,Umin,Umax)
    A = [Am, zeros(n,q); Cm*Am, eye(q)];
    B = [Bm; Cm*Bm];
    C = [zeros(q,n), eye(q,q)];
    
    %Y = F*x0 + phi*delU
    
    %defining F
    F = [];
    for i=1:Np
        F = [F;C*A^i];
    end
    
    %defining phi
    phi = [];
    for i=1:Nc
        col = zeros(q*(i-1),m);
        for j=i:Np
            col = [col;C*A^(j-i)*B];
        end
        phi = [phi,col];
    end

    R = eye(m*Nc);

    %Defining the set point
    Rs = [];
    for i=1:Np
        Rs = [Rs;Rs0];
    end

    %Defining Constraints
    %Amplitude of Control Signals
    C1 = [];
    for i=1:Nc
        C1 = [C1;eye(m)];
    end

    C2 = [];
    for i=1:Nc
        col = zeros(m*(i-1),m);
        for j=i:Nc
            col = [col;eye(m)];
        end
        C2 = [C2,col];
    end

    UMIN = [];
    for i=1:Nc
        UMIN = [UMIN;Umin];
    end

    UMAX = [];
    for i=1:Nc
        UMAX = [UMAX;Umax];
    end

    
    N1 = [-UMIN + C1*u;UMAX - C1*u];
    M1 = [-C2;C2];

    %Incremental Changes in Inputs
    DUMIN = [];
    for i=1:Nc
        DUMIN = [DUMIN;DUmin];
    end

    DUMAX = [];
    for i=1:Nc
        DUMAX = [DUMAX;DUmax];
    end


    N2 = [-DUMIN;DUMAX];
    M2 = [eye(Nc*m);-eye(Nc*m)];


    E = phi'*phi+R; %Hessian
    G = -phi'*(Rs - F*x0);

    delU = quadprog(E,G',[M1;M2],[N1;N2]);
    

end

function [delU,F,phi,Ru,A,B] = controlAction(Np,Nc,n,q,m,x0,Rs0,Am,Bm,Cm)
    
    A = [Am, zeros(n,q); Cm*Am, eye(q)];
    B = [Bm; Cm*Bm];
    C = [zeros(q,n), eye(q,q)];
    
    %Y = F*x0 + phi*delU
    
    %defining F
    F = [];
    for i=1:Np
        F = [F;C*A^i];
    end
    
    %defining phi
    phi = [];
    for i=1:Nc
        col = zeros(q*(i-1),m);
        for j=i:Np
            col = [col;C*A^(j-i)*B];
        end
        phi = [phi,col];
    end

    
    
    %defining the set point
    Ru = eye(m*Nc);
    Rs = [];
    for i=1:Np
        Rs = [Rs;Rs0];
    end
    
    
    
    delU = inv(phi'*phi+Ru)*phi'*(Rs - F*x0);
    
    
end


