WorkingPath = 'C:\Users\zinya\Projects\2dMHDProject\outputs\20240308_105944\';

N = 512;

for i=0:0
    floderName = WorkingPath;
    if (i < 10)
       floderName = [floderName, '00', int2str(i)];
    end
    
    if ((i >= 10)&&(i < 100))
       floderName = [floderName, '0', int2str(i)];
    end
    
    if (i >= 100)
        floderName = [floderName, int2str(i)];
    end
    
    fileStream = fopen([floderName,'\stream'], 'rb');
    Stream = fread(fileStream, [N N], 'double');
    fclose(fileStream);
    
    pcolor(Stream);
    shading flat; 
    
    pause(0.001);
end
