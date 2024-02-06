function [ Z, E, k ] = spectrum_term( w, tau, varargin )
N = size(w,1);
L = 2*pi;
lmax = floor(sqrt((N/2)^2 + (N/2)^2));
kx = [0:N/2 (-N/2 + 1):-1];
ky = kx;
wf = fft2(w);
tauf = fft2(tau);
Z = zeros(lmax,1);
if (length(varargin)>0)
    dt = varargin{1};
    for m=1:N
        for n=1:N
            K=sqrt( kx(n) * kx(n) + ky(m) * ky(m) );
            l=floor(K);
            if (l>0)
                Z(l)=Z(l) + real(conj(wf(n,m)) * tauf(n,m)) + dt/2*conj(tauf(n,m))*tauf(n,m);
            end
        end
    end
else
    for m=1:N
        for n=1:N
            K=sqrt( kx(n) * kx(n) + ky(m) * ky(m) );
            l=floor(K);
            if (l>0)
                Z(l)=Z(l) + real(conj(wf(n,m)) * tauf(n,m));
            end
        end
    end
    
end
Z = Z * L^2 / (2*N^4);
k = 1.5:1:lmax+0.5;
k=k';
E = Z ./ k.^2;
end