function Static_Dataset
    E = 83e9;
    pratio=0.33;
    G = E/(2*(1+pratio)); % have to find correct value
    pho = 6450;
    g = [0;0;9.81];
    rad = 0.0025;
    tendonOffset = 0.02;
    L = 0.34;
    noTendons = 4;

    Area = pi*rad^2;
    I = pi*rad^4/4;
    J = 2*I;
    Kse = diag([G*Area,G*Area,E*Area]);
    Kbt = diag([E*I,E*I,E*J]);
    initial_guess = zeros(6,1);
    tau = [1;1;1;1];
    p0 = [0;0;0];
    R0 = eye(3);
    r = [tendonDistance(0) tendonDistance(pi/2) tendonDistance(pi) tendonDistance(3*pi/2)];
    global Y
    fsolve(@staticCalculation,initial_guess)
    
    function y_s = CosseratStaticModel(s,y)
        R = reshape(y(4:12),3,3);
        u = y(16:18);
        v = y(13:15);
        
        n = Kse*(v-[0;0;1]);
        m = Kbt*u;
        
        A = zeros(3,3);
        B = zeros(3,3);
        Gv = zeros(3,3);
        H = zeros(3,3);
        a = zeros(3,1);
        b = zeros(3,1);
        
        for i=1:noTendons
            p_b =hat(u)*r(:,i)+v;
            p_bnorm = norm(p_b);
            Ai=-tau(i)*(hat(p_b)^2)/(p_bnorm)^3;
            Bi = hat(r(:,i))*Ai;
            Gi = -Ai*hat(r(:,i));
            Hi = -Bi*hat(r(:,i));
            ai = Ai*(hat(u)*p_b);
            bi = hat(r(:,i))*ai;
            A = A + Ai;
            B = B + Bi;
            Gv = Gv + Gi;
            H = H + Hi;
            a = a + ai;
            b = b + bi;
        end
        
        f = pho*Area*g;
        KSBT = [Kse+A Gv;
                B Kbt+H];
        rhs = [-hat(u)*n-R.'*f-a;
               -hat(u)*m-hat(v)*n-b];
        
        v_u_sys = KSBT\rhs;
        p = R*v;
        Rdot = R*hat(u);
        
        y_s = [p;reshape(Rdot,9,1);v_u_sys];
    end

    function distal_error = staticCalculation(guess)
        n0 = guess(1:3);
        u0 = guess(4:6);
        v0 = Kse\n0 + [0;0;1];
        
        y0 = [p0;reshape(R0,9,1);v0;u0];
        [~,Y] = ode45(@CosseratStaticModel,linspace(0,L),y0);
        disp(Y(end,1:3))
        vL = Y(end,13:15).';
        uL = Y(end,16:18).';
        
        nb = Kse*(vL - [0;0;1]);
        mb = Kbt*uL;
        
        %Find the equilibrium error at the tip, considering tendon forces
        force_error = -nb;
        moment_error = -mb;
        for i = 1 : noTendons
            pb_si = cross(uL,r(:,i)) + vL;
            Fb_i = -tau(i)*pb_si/norm(pb_si);
            force_error = force_error + Fb_i;
            moment_error = moment_error + cross(r(:,i), Fb_i);
        end
        
        distal_error = [force_error; moment_error];
    end

    function skew_mat = hat(m)
        
        skew_mat = [0 -m(3) m(2);
                    m(3) 0 -m(1);
                    -m(2) m(1) 0];
    end
    
    function dist = tendonDistance(theta)
        dist = tendonOffset*[cos(theta);sin(theta);0];
    end
end
