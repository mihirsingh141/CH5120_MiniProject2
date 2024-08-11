clc;
clear;
close all;


%Initial Values
global A1 A2 A3 A4 a1 a2 a3 a4 kc g k1 k2 v1 v2 gamma1 gamma2 Ts

A1 = 28; %(cm^2)
A2 = 32;
A3 = 28;
A4 = 32;

a1 = 0.071; a3 = 0.071; %(cm^2)
a2 = 0.057; a4 = 0.057;

kc = 0.5; %(V/cm)
g = 981; %(cm/s^2)

gamma1 = 0.7; gamma2 = 0.6; %constants that are determined from valve postion

k1 = 3.33; k2 = 3.35; %(cm^3/Vs)

v1 = 3;v2 = 3; %(V)

h0 = [12.4; 12.7; 1.8; 1.4];



Cc = kc*[1,0,0,0;0,1,0,0];
Cd = Cc;

u = [v1;v2];


%Ordinary Differential Equation System

tspan = linspace(0,400,10000);
    
[t,hs] = ode45(@ODE,tspan,h0);
y = Cc*hs' + 0.001*eye(2)*randn(2,10000);

%Extended Kalman Filter Application

Ts = tspan(2)-tspan(1);


x_prior = zeros(4,10001);
x_posterior = zeros(4,10001);

Z_est_prior = zeros(2,10001);
Z_est_posterior = zeros(2,10001);

x0 = 0.95*h0;  
x_posterior(:,1) = x0;
x_prior(:,1) = x0;

P_prior = zeros(4, 4, 10001);
P_posterior = zeros(4,4,10001);
K = zeros(4, 2, 10001);

P0 = eye(4); %arbitrary initial state error covariance matrix
P_posterior(:,:,1) = P0;

resid_prior = zeros(4,10001);
resid_posterior = zeros(4,10001);

Q = 0.00001*eye(4); %covariance matrix of noise in state equation
R = 0.00001*eye(2); %covariance matrix of noise in measurement equation



for k=1:10000
    %Prediction

    x_prior(:,k+1) = non_lin_pred(x_posterior(:,k));

    if k<=10
        A = A_pred(x_posterior(:,k));
    end

    P_prior(:,:,k+1) = A*P_posterior(:,:,k)*A'+ Q;

    
    resid_prior(:,k+1) = hs(k,:)' - x_prior(:,k+1);
    
    %Updation
    
    K(:,:,k+1) = P_prior(:,:,k+1)*Cd'*inv(Cd*P_prior(:,:,k+1)*Cd' + R);
    x_posterior(:,k+1) = x_prior(:,k+1) + K(:,:,k+1)*(y(:,k) - Cd*(x_prior(:,k+1)));
    P_posterior(:,:,k+1) = (eye(4) - K(:,:,k+1)*Cd)*P_prior(:,:,k+1);

    resid_posterior(:,k+1) = hs(k,:)' - x_posterior(:,k+1);
    

end


%Results visualisation

k_span = 1:10000;

%Posterior Residuals
figure;
plot(k_span, resid_posterior(1,2:10001), 'LineWidth', 2);
hold on

plot(k_span, resid_posterior(2,2:10001), 'LineWidth', 2);
hold on

plot(k_span, resid_posterior(3,2:10001), 'LineWidth', 2);
hold on

plot(k_span, resid_posterior(4,2:10001), 'LineWidth', 2);
hold on

legend('h1','h2','h3','h4')
title('Plots of posterior residuals')


%Prior Residuals
figure;
plot(k_span, resid_prior(1,2:10001), 'LineWidth', 2);
hold on

plot(k_span, resid_prior(2,2:10001), 'LineWidth', 2);
hold on

plot(k_span, resid_prior(3,2:10001), 'LineWidth', 2);
hold on

plot(k_span, resid_prior(4,2:10001), 'LineWidth', 2);
hold on

legend('h1','h2','h3','h4')
title('Plots of prior residuals')

%Prior Covariance
figure;

subplot(4,4,1);
plot(k_span,squeeze(P_prior(1,1,2:10001)),'LineWidth', 2);
subplot(4,4,2);
plot(k_span,squeeze(P_prior(1,2,2:10001)),'LineWidth', 2);
subplot(4,4,3);
plot(k_span,squeeze(P_prior(1,3,2:10001)),'LineWidth', 2);
subplot(4,4,4);
plot(k_span,squeeze(P_prior(1,4,2:10001)),'LineWidth', 2);

subplot(4,4,5);
plot(k_span,squeeze(P_prior(2,1,2:10001)),'LineWidth', 2);
subplot(4,4,6);
plot(k_span,squeeze(P_prior(2,2,2:10001)),'LineWidth', 2);
subplot(4,4,7);
plot(k_span,squeeze(P_prior(2,3,2:10001)),'LineWidth', 2);
subplot(4,4,8);
plot(k_span,squeeze(P_prior(2,4,2:10001)),'LineWidth', 2);


subplot(4,4,9);
plot(k_span,squeeze(P_prior(3,1,2:10001)),'LineWidth', 2);
subplot(4,4,10);
plot(k_span,squeeze(P_prior(3,2,2:10001)),'LineWidth', 2);
subplot(4,4,11);
plot(k_span,squeeze(P_prior(3,3,2:10001)),'LineWidth', 2);
subplot(4,4,12);
plot(k_span,squeeze(P_prior(3,4,2:10001)),'LineWidth', 2);

subplot(4,4,13);
plot(k_span,squeeze(P_prior(4,1,2:10001)),'LineWidth', 2);
subplot(4,4,14);
plot(k_span,squeeze(P_prior(4,2,2:10001)),'LineWidth', 2);
subplot(4,4,15);
plot(k_span,squeeze(P_prior(4,3,2:10001)),'LineWidth', 2);
subplot(4,4,16);
plot(k_span,squeeze(P_prior(4,4,2:10001)),'LineWidth', 2);

sgtitle('Prior Covariances')



%Posterior Covariances
figure;

subplot(4,4,1);
plot(k_span,squeeze(P_posterior(1,1,2:10001)),'LineWidth', 2);
subplot(4,4,2);
plot(k_span,squeeze(P_posterior(1,2,2:10001)),'LineWidth', 2);
subplot(4,4,3);
plot(k_span,squeeze(P_posterior(1,3,2:10001)),'LineWidth', 2);
subplot(4,4,4);
plot(k_span,squeeze(P_posterior(1,4,2:10001)),'LineWidth', 2);

subplot(4,4,5);
plot(k_span,squeeze(P_posterior(2,1,2:10001)),'LineWidth', 2);
subplot(4,4,6);
plot(k_span,squeeze(P_posterior(2,2,2:10001)),'LineWidth', 2);
subplot(4,4,7);
plot(k_span,squeeze(P_posterior(2,3,2:10001)),'LineWidth', 2);
subplot(4,4,8);
plot(k_span,squeeze(P_posterior(2,4,2:10001)),'LineWidth', 2);


subplot(4,4,9);
plot(k_span,squeeze(P_posterior(3,1,2:10001)),'LineWidth', 2);
subplot(4,4,10);
plot(k_span,squeeze(P_posterior(3,2,2:10001)),'LineWidth', 2);
subplot(4,4,11);
plot(k_span,squeeze(P_posterior(3,3,2:10001)),'LineWidth', 2);
subplot(4,4,12);
plot(k_span,squeeze(P_posterior(3,4,2:10001)),'LineWidth', 2);

subplot(4,4,13);
plot(k_span,squeeze(P_posterior(4,1,2:10001)),'LineWidth', 2);
subplot(4,4,14);
plot(k_span,squeeze(P_posterior(4,2,2:10001)),'LineWidth', 2);
subplot(4,4,15);
plot(k_span,squeeze(P_posterior(4,3,2:10001)),'LineWidth', 2);
subplot(4,4,16);
plot(k_span,squeeze(P_posterior(4,4,2:10001)),'LineWidth', 2);

sgtitle('Posterior Covariances')




%Kalman Filters
figure;
subplot(4,2,1);
plot(k_span,squeeze(K(1,1,2:10001)),'LineWidth', 2);
subplot(4,2,2);
plot(k_span,squeeze(K(2,1,2:10001)),'LineWidth', 2);
subplot(4,2,3)
plot(k_span,squeeze(K(3,1,2:10001)),'LineWidth', 2);
subplot(4,2,4);
plot(k_span,squeeze(K(4,1,2:10001)),'LineWidth', 2);

subplot(4,2,5);
plot(k_span,squeeze(K(1,2,2:10001)),'LineWidth', 2);
subplot(4,2,6);
plot(k_span,squeeze(K(2,2,2:10001)),'LineWidth', 2);
subplot(4,2,7);
plot(k_span,squeeze(K(3,2,2:10001)),'LineWidth', 2);
subplot(4,2,8);
plot(k_span,squeeze(K(4,2,2:10001)),'LineWidth', 2);
sgtitle('Kalman Filters')

%h3 and h4
figure;
subplot(2,1,1)
plot(k_span, x_posterior(3,2:10001), k_span, hs(:,3), 'LineWidth', 2);
legend('Estimated h3','Simulated h3')
title('h3 estimates')
hold on

subplot(2,1,2)
plot(k_span, x_posterior(4,2:10001), k_span, hs(:,4), 'LineWidth', 2);
legend('Estimated h4','Simulated h4')
title('h4 estimtates')
hold on



function f = A_pred(x)
    global A1 A2 A3 A4 a1 a2 a3 a4 Ts g
    
    f = zeros(4);
    f(1,1) = 1  - ((2*g)^0.5)*Ts*a1/(2*A1*x(1)^0.5);
    f(1,3) = ((2*g)^0.5)*Ts*a3/(2*A1*x(3)^0.5);
    f(2,2) = 1  - ((2*g)^0.5)*Ts*a2/(2*A2*x(2)^0.5);
    f(2,4) = ((2*g)^0.5)*Ts*a4/(2*A2*x(4)^0.5);
    f(3,3) = 1  - ((2*g)^0.5)*Ts*a3/(2*A3*x(3)^0.5);
    f(4,4) = 1  - ((2*g)^0.5)*Ts*a4/(2*A4*x(4)^0.5);
    
end

function f = non_lin_pred(x)

    global A1 A2 A3 A4 a1 a2 a3 a4 g k1 k2 v1 v2 gamma1 gamma2 Ts
    
    
    f = zeros(4,1);
    f(1) = x(1) + Ts*((-a1/A1)*sqrt(2*g*x(1)) + (a3/A1)*sqrt(2*g*x(3)) + gamma1*k1*v1/A1);
    f(2) = x(2) + Ts*((-a2/A2)*sqrt(2*g*x(2)) + (a4/A2)*sqrt(2*g*x(4)) + gamma2*k2*v2/A2);
    f(3) = x(3) + Ts*((-a3/A3)*sqrt(2*g*x(3)) + (1-gamma2)*k2*v2/A3);
    f(4) = x(4) + Ts*((-a4/A4)*sqrt(2*g*x(4)) + (1-gamma1)*k1*v1/A4);
    
end




function dhdt = ODE(t,h)
    global A1 A2 A3 A4 a1 a2 a3 a4 kc g gamma1 gamma2 k1 k2 v1 v2
    

    dhdt = zeros(4,1);

    dhdt(1) = -a1/A1*(2*g*h(1))^0.5 + a3/A1*(2*g*h(3))^0.5 + gamma1*k1/A1*v1;
    dhdt(2) = -a2/A2*(2*g*h(2))^0.5 + a4/A2*(2*g*h(4))^0.5 + gamma2*k2/A2*v2;
    dhdt(3) = -a3/A3*(2*g*h(3))^0.5 + (1-gamma2)*k2*v2/A3;
    dhdt(4) = -a4/A4*(2*g*h(4))^0.5 + (1-gamma1)*k1*v1/A4;
end
