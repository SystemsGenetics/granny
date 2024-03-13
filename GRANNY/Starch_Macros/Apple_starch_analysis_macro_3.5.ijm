//fresh start, applying appropriate settings
close("*");
run("Clear Results");
roiManager("reset");
selectWindow("ROI Manager");
run("Close");
print("\\Clear");
selectWindow("Log");
run("Close");
setOption("BlackBackground", true);
run("Set Measurements...", "display area redirect=None decimal=3");
run("Colors...", "foreground=white background=black selection=yellow"); 
run("Line Width...", "line=1");

//choosing input and output directories
dir = getDirectory("Choose input folder");
output_path = getDirectory("Choose output folder"); 
fileList = getFileList(output_path); 
list = getFileList(dir);
print("Cross-Section"+","+"Total cross-section area"+","+"Starch area #1"+","+"Starch area #2"+","+"Starch area #3"+","+"Starch area #4"+","+"Starch area #5"+","+"Starch area #6"+","+"Starch area #7"+","+"Starch area #8"+","+"Starch area #9"+","+"Starch area #10"+","+"Total Starch Area"+","+"Percent Starch Area"+","+"Cornell Starch Rating"+","+"Jonagold Starch Rating"+","+"Purdue Starch Rating"+","+"UC GrannySmith Starch Rating");

//enabling batch mode of all images of input folder
setBatchMode(true); 
for (i = 0; i < list.length; i++){   
    processImage(dir,list[i]);
}

function processImage(dir,image){
    open(dir+image);
    fileNoExtension = File.nameWithoutExtension;

//macro to isolate starch from threshold images    
run("8-bit");
run("Manual Threshold...", "min=5 max=172"); //threshold of starch, adjust if necessary
makeLine(196, 0, 196, 500);
run("Fill", "slice"); //draws division line in-case starch area runs the full circumference of the cross section
run("Analyze Particles...", "size=250-Infinity display"); //outputs starch area 
saveAs(".jpg",output_path+image+"starch.jpg"); //saves starch threshold images
wait(200);

// if / then scripts based on number of identified regions of starch 
if (nResults == 1){
C = getResult("Area", 0);
}
if (nResults == 2){
C = getResult("Area", 0);
D = getResult("Area", 1);
}
if (nResults == 3){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
}
if (nResults == 4){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
}
if (nResults == 5){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
G = getResult("Area", 4);
}
if (nResults == 6){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
G = getResult("Area", 4);
H = getResult("Area", 5);
}
if (nResults == 7){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
G = getResult("Area", 4);
H = getResult("Area", 5);
I = getResult("Area", 6);
}
if (nResults == 8){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
G = getResult("Area", 4);
H = getResult("Area", 5);
I = getResult("Area", 6);
J = getResult("Area", 7);
}
if (nResults == 9){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
G = getResult("Area", 4);
H = getResult("Area", 5);
I = getResult("Area", 6);
J = getResult("Area", 7);
K = getResult("Area", 8);
}
if (nResults == 10){ 
C = getResult("Area", 0);
D = getResult("Area", 1);
E = getResult("Area", 2);
F = getResult("Area", 3);
G = getResult("Area", 4);
H = getResult("Area", 5);
I = getResult("Area", 6);
J = getResult("Area", 7);
K = getResult("Area", 8);
L = getResult("Area", 9);
}

wait(200);
run("Manual Threshold...", "min=5 max=255"); //threshold of the total cross-section area, adjust if necessary
run("Analyze Particles...", "size=2000-Infinity display"); //outputs total area
saveAs(".jpg",output_path+image+"total_area.jpg"); //saves total cross-section threshold images
wait(200);
if (nResults > 0){
	B = getResult("Area", nResults-1);
}

// if / then scripts to output data in comma delmitted format 
if (nResults == 1){
print(fileNoExtension+","+B+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0");
}
if (nResults == 2){
print(fileNoExtension+","+B+","+C+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+C+","+C/B*100+","+(round(((C/B*100)*(C/B*100)*-0.0006-0.0056*(C/B*100)+7.5874)))+","+(round(((C/B*100)*(C/B*100)*-0.0006-0.0371*(C/B*100)+9.2883)))+","+(round(((C/B*100)*(C/B*100)*-0.0003-0.0306*(C/B*100)+5.9517)))+","+(round(((C/B*100)*(C/B*100)*-0.0006+0.001*(C/B*100)+5.7876))));
}
if (nResults == 3){ 
print(fileNoExtension+","+B+","+C+","+D+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+C+D+","+((C+D)/B*100)+","+(round((((C+D)/B*100)*((C+D)/B*100)*-0.0006-0.0056*((C+D)/B*100)+7.5874)))+","+(round((((C+D)/B*100)*((C+D)/B*100)*-0.0006-0.0371*((C+D)/B*100)+9.2883)))+","+(round((((C+D)/B*100)*((C+D)/B*100)*-0.0003-0.0306*((C+D)/B*100)+5.9517)))+","+(round((((C+D)/B*100)*((C+D)/B*100)*-0.00066+0.001*((C+D)/B*100)+5.7876))));
}
if (nResults == 4){ 
print(fileNoExtension+","+B+","+C+","+D+","+E+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+C+D+E+","+((C+D+E)/B*100)+","+(round((((C+D+E)/B*100)*((C+D+E)/B*100)*-0.0006-0.0056*((C+D+E)/B*100)+7.5874)))+","+(round((((C+D+E)/B*100)*((C+D+E)/B*100)*-0.0006-0.0371*((C+D+E)/B*100)+9.2883)))+","+(round((((C+D+E)/B*100)*((C+D+E)/B*100)*-0.0003-0.0306*((C+D+E)/B*100)+5.9517)))+","+(round((((C+D+E)/B*100)*((C+D+E)/B*100)*-0.00066+0.001*((C+D+E)/B*100)+5.7876))));
}
if (nResults == 5){ 
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+C+D+E+F+","+((C+D+E+F)/B*100)+","+(round((((C+D+E+F)/B*100)*((C+D+E+F)/B*100)*-0.0006-0.0056*((C+D+E+F)/B*100)+7.5874)))+","+(round((((C+D+E+F)/B*100)*((C+D+E+F)/B*100)*-0.0006-0.0371*((C+D+E+F)/B*100)+9.2883)))+","+(round((((C+D+E+F)/B*100)*((C+D+E+F)/B*100)*-0.0003-0.0306*((C+D+E+F)/B*100)+5.9517)))+","+(round((((C+D+E+F)/B*100)*((C+D+E+F)/B*100)*-0.00066+0.001*((C+D+E+F)/B*100)+5.7876))));
}
if (nResults == 6){ 
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+G+","+"0"+","+"0"+","+"0"+","+"0"+","+"0"+","+C+D+E+F+G+","+((C+D+E+F+G)/B*100)+","+(round((((C+D+E+F+G)/B*100)*((C+D+E+F+G)/B*100)*-0.0006-0.0056*((C+D+E+F+G)/B*100)+7.5874)))+","+(round((((C+D+E+F+G)/B*100)*((C+D+E+F+G)/B*100)*-0.0006-0.0371*((C+D+E+F+G)/B*100)+9.2883)))+","+(round((((C+D+E+F+G)/B*100)*((C+D+E+F+G)/B*100)*-0.0003-0.0306*((C+D+E+F+G)/B*100)+5.9517)))+","+(round((((C+D+E+F+G)/B*100)*((C+D+E+F+G)/B*100)*-0.00066+0.001*((C+D+E+F+G)/B*100)+5.7876))));
}
if (nResults == 7){
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+G+","+H+","+"0"+","+"0"+","+"0"+","+"0"+","+C+D+E+F+G+H+","+((C+D+E+F+G+H)/B*100)+","+(round((((C+D+E+F+G+H)/B*100)*((C+D+E+F+G+H)/B*100)*-0.0006-0.0056*((C+D+E+F+G+H)/B*100)+7.5874)))+","+(round((((C+D+E+F+G+H)/B*100)*((C+D+E+F+G+H)/B*100)*-0.0006-0.0371*((C+D+E+F+G+H)/B*100)+9.2883)))+","+(round((((C+D+E+F+G+H)/B*100)*((C+D+E+F+G+H)/B*100)*-0.0003-0.0306*((C+D+E+F+G+H)/B*100)+5.9517)))+","+(round((((C+D+E+F+G+H)/B*100)*((C+D+E+F+G+H)/B*100)*-0.00066+0.001*((C+D+E+F+G+H)/B*100)+5.7876))));
}
if (nResults == 8){
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+G+","+H+","+I+","+"0"+","+"0"+","+"0"+","+C+D+E+F+G+H+I+","+((C+D+E+F+G+H+I)/B*100)+","+(round((((C+D+E+F+G+H+I)/B*100)*((C+D+E+F+G+H+I)/B*100)*-0.0006-0.0056*((C+D+E+F+G+H+I)/B*100)+7.5874)))+","+(round((((C+D+E+F+G+H+I)/B*100)*((C+D+E+F+G+H+I)/B*100)*-0.0006-0.0371*((C+D+E+F+G+H+I)/B*100)+9.2883)))+","+(round((((C+D+E+F+G+H+I)/B*100)*((C+D+E+F+G+H+I)/B*100)*-0.0003-0.0306*((C+D+E+F+G+H+I)/B*100)+5.9517)))+","+(round((((C+D+E+F+G+H+I)/B*100)*((C+D+E+F+G+H+I)/B*100)*-0.00066+0.001*((C+D+E+F+G+H+I)/B*100)+5.7876))));
}
if (nResults == 9){
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+G+","+H+","+I+","+J+","+"0"+","+"0"+","+C+D+E+F+G+H+I+J+","+((C+D+E+F+G+H+I+J)/B*100)+","+(round((((C+D+E+F+G+H+I+J)/B*100)*((C+D+E+F+G+H+I+J)/B*100)*-0.0006-0.0056*((C+D+E+F+G+H+I+J)/B*100)+7.5874)))+","+(round((((C+D+E+F+G+H+I+J)/B*100)*((C+D+E+F+G+H+I+J)/B*100)*-0.0006-0.0371*((C+D+E+F+G+H+I+J)/B*100)+9.2883)))+","+(round((((C+D+E+F+G+H+I+J)/B*100)*((C+D+E+F+G+H+I+J)/B*100)*-0.0003-0.0306*((C+D+E+F+G+H+I+J)/B*100)+5.9517)))+","+(round((((C+D+E+F+G+H+I+J)/B*100)*((C+D+E+F+G+H+I+J)/B*100)*-0.00066+0.001*((C+D+E+F+G+H+I+J)/B*100)+5.7876))));
}
if (nResults == 10){
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+G+","+H+","+I+","+J+","+K+","+"0"+","+C+D+E+F+G+H+I+J+K+","+((C+D+E+F+G+H+I+J+K)/B*100)+","+(round((((C+D+E+F+G+H+I+J+K)/B*100)*((C+D+E+F+G+H+I+J+K)/B*100)*-0.0006-0.0056*((C+D+E+F+G+H+I+J+K)/B*100)+7.5874)))+","+(round((((C+D+E+F+G+H+I+J+K)/B*100)*((C+D+E+F+G+H+I+J+K)/B*100)*-0.0006-0.0371*((C+D+E+F+G+H+I+J+K)/B*100)+9.2883)))+","+(round((((C+D+E+F+G+H+I+J+K)/B*100)*((C+D+E+F+G+H+I+J+K)/B*100)*-0.0003-0.0306*((C+D+E+F+G+H+I+J+K)/B*100)+5.9517)))+","+(round((((C+D+E+F+G+H+I+J+K)/B*100)*((C+D+E+F+G+H+I+J+K)/B*100)*-0.00066+0.001*((C+D+E+F+G+H+I+J+K)/B*100)+5.7876))));
}
if (nResults == 11){
print(fileNoExtension+","+B+","+C+","+D+","+E+","+F+","+G+","+H+","+I+","+J+","+K+","+L+","+C+D+E+F+G+H+I+J+K+L+","+((C+D+E+F+G+H+I+J+K+L)/B*100)+","+(round((((C+D+E+F+G+H+I+J+K+L)/B*100)*((C+D+E+F+G+H+I+J+K+L)/B*100)*-0.0006-0.0056*((C+D+E+F+G+H+I+J+K+L)/B*100)+7.5874)))+","+(round((((C+D+E+F+G+H+I+J+K+L)/B*100)*((C+D+E+F+G+H+I+J+K+L)/B*100)*-0.0006-0.0371*((C+D+E+F+G+H+I+J+K+L)/B*100)+9.2883)))+","+(round((((C+D+E+F+G+H+I+J+K+L)/B*100)*((C+D+E+F+G+H+I+J+K+L)/B*100)*-0.0003-0.0306*((C+D+E+F+G+H+I+J+K+L)/B*100)+5.9517)))+","+(round((((C+D+E+F+G+H+I+J+K+L)/B*100)*((C+D+E+F+G+H+I+J+K+L)/B*100)*-0.00066+0.001*((C+D+E+F+G+H+I+J+K+L)/B*100)+5.7876))));
}
close("Results");
}
selectWindow("Log");
saveAs("text",output_path+"Results.csv"); //saves the log window as the final results.csv file 
close("Log")


