WorkingPath = 'data\20240205_215905\';

N = 2048;

for i=0:41
    fileName = 'out';
    if (i < 10)
       fileName = [fileName, '00', int2str(i)];
    end
    
    if ((i >= 10)&&(i < 100))
       fileName = [fileName, '0', int2str(i)];
    end
    
    if (i >= 100)
        fileName = [fileName, int2str(i)];
    end
    
    fileOmega = fopen([WorkingPath, 'vorticity\', fileName], 'rb');
    Omega = fread(fileOmega, [N N], 'double');
    fclose(fileOmega);
    
    filePsi = fopen([WorkingPath, 'streamFunction\', fileName], 'rb');
    Psi = fread(filePsi, [N N], 'double');
    fclose(filePsi);

    E_kin = abs(spectrum_term(Omega, Psi));

    fileJ = fopen([WorkingPath, 'current\', fileName], 'rb');
    J = fread(fileJ, [N N], 'double');
    fclose(fileJ);

    fileA = fopen([WorkingPath, 'magneticPotential\', fileName], 'rb');
    A = fread(fileA, [N N], 'double');
    fclose(fileA);
  
    E_mag = abs(spectrum_term(J, A));

    E = E_kin + E_mag;

    loglog(E(1:900));
    pause(0.001);
end
