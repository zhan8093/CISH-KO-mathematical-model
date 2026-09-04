function [sol] = cish_model_fun_v2(vars_all, y0, cond)
    % Time points
    t0 = 0;
    t_end = vars_all.t_end;
    
    % Doses (now become initial conditions for dose_L and dose_Lo)
    dose_L = cond(1);
    dose_Lo = cond(2);
    dose_Z = cond(3);
    
    % Set initial conditions with dose compartments
    y0_modified = y0;
    y0_modified(6) = dose_L;   % dose_L(0)
    y0_modified(7) = dose_Lo;  % dose_Lo(0)
    
    % Apply instantaneous dose to Z only (injected directly)
    y0_modified(5) = y0_modified(5) + dose_Z;
    
    % Solve from t0 to t_end (single period)
    [t_total, y_total] = ode45(@(t, y) ode_rhs(t, y, vars_all), [t0, t_end], y0_modified);
    
    % Output
    sol.x = t_total(:)';   % Force row vector
    sol.y = y_total';       % Already (nStates × nPoints)
end

function dydt = ode_rhs(t, y, p)
    % Unpack state variables
    T  = max(y(1), 1e-6);
    L  = max(y(2), 1e-6);
    Lo = max(y(3), 1e-6);
    N  = max(y(4), 1e-6);
    Z  = max(y(5), 1e-6);
    dose_L  = max(y(6), 1e-6);
    dose_Lo = max(y(7), 1e-6);

    % Unpack parameters
    K = p.K;
    gamma = p.gamma;
    muL = p.muL;
    muN = p.muN;
    muZ = p.muZ;
    dL_param = p.dL;      % RENAMED
    dN_param = p.dN;      % RENAMED
    pL = p.pL;
    pN = p.pN;
    delta = p.delta;
    h = p.h;
    a = p.a;
    b = p.b;
    bL = p.bL;
    kL = p.kL;
    kLo = p.kLo;
    kN = p.kN;
    kappa = p.kappa;

    % Derived terms
    alpha = (a + 1) ./ (a + exp(-b * Z));
    alpha_L = (a + 1) ./ (a + exp(-bL * Z));
    capacity = 1 - (L + Lo + N) / K;

    % ODE system
    dTdt = gamma * T ...
        - alpha_L .* kL  .* L  .* T ...
        - alpha   .* kLo .* Lo .* T ...
        - alpha   .* kN  .* N  .* T;

    dZdt = -muZ * Z;

    dLdt = -muL * L * T / (1 + Z) ...
         - dL_param * L ...
         + capacity * pL * L * (1 + Z) ...
         + kappa * dose_L;

    dLodt = -muN * Lo * T / (1 + Z) ...
          - dN_param * Lo ...
          + capacity * pN * Lo * (1 + Z) ...
          + kappa * dose_Lo;

    dNdt = -muN * N * T / (1 + Z) ...
         - dN_param * N ...
         + capacity * pN * N * (1 + Z) ...
         + capacity * delta * T / (h + T);

    ddose_Ldt  = -kappa * dose_L;
    ddose_Lodt = -kappa * dose_Lo;

    dydt = [dTdt; dLdt; dLodt; dNdt; dZdt; ddose_Ldt; ddose_Lodt];
end