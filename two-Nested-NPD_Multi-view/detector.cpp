#include <detector.h>

/*
//________________LIBRERIAS OPENCV___________________
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//___________________________________________________
*/

//________________________VARIABLES GLOBALES AL ARCHIVO______________________________________________//

static const int pixelResolution=256;//Resolución de píxeles, de esto depende los valores posibles de la evaluación de una NPD

static float NPD_VALUES_FOR_EVALUATION[pixelResolution][pixelResolution];//LOCK-TABLE adecuada para la evaluación

bool generateLockTableEvaluationDetector()//Aquí llenamos NPD_VALUES_FOR_EVALUATION
{

float npd_temp;
for(int x=0;x<pixelResolution;x++)
{
for(int y=0;y<pixelResolution;y++)
{
npd_temp=(float(x-y))/(x+y);
if(npd_temp!=npd_temp)npd_temp=0;//detectando valores NaN
NPD_VALUES_FOR_EVALUATION[x][y]=npd_temp;
}
}

return true;
}

static bool flagFill=generateLockTableEvaluationDetector();//Esto es para que NPD_VALUES_FOR_EVALUATION se llene al iniciar el programa

//____________________________________________________________________________________________________//



class WINDOWING
{
cv::Point2f center;
double degree;
int scale;
public:
WINDOWING(cv::Point2f  c,int s,double degree):center(c),scale(s),degree(degree){}

};





NODE_EVALUATION::~NODE_EVALUATION()
{


if(!nodeIsTerminal)
{

//______________Pila donde se almacenan las características rotadas y escaladas_______________//
if(!stackFeatures.empty()){
for(int i=0;i<stackFeatures.size();i++)
{

for(int j=0;j<stackFeatures[i].size();j++){
delete []stackFeatures[i][j];
}

}
stackFeatures.clear();
}
//____________________________________________________________________________________________//

delete feature;
delete nodeLeft;
delete nodeRight;
}

}


void NODE_EVALUATION::setNodeAsTerminal()
{
pointerToFunctionEvaluation=&NODE_EVALUATION::evaluationNodeTerminal;
nodeIsTerminal=true;
}


void NODE_EVALUATION::loadNode(cv::FileNode nodeRootFile)
{

cv::FileNode information=nodeRootFile["information"];
nodeIsTerminal=(bool)(int)information["nodeIsTerminal"];
threshold=(double)information["threshold"];
numFeature=(int)information["numFeature"];


if(!nodeIsTerminal){
feature=new cv::Point2i;
*feature=parentClassifier->NPD[numFeature];
}



if(nodeIsTerminal)setNodeAsTerminal();/*Debe llamarse si el nodo es terminal*/ 

yt=(double)information["yt"];

/*_________________APLICAMOS LA RECURSIVIDAD__________________________*/
if(!nodeIsTerminal){

cv::FileNode nodeLeftFile=nodeRootFile["nodeLeft"];
nodeLeft=new NODE_EVALUATION(this,parentClassifier);
nodeLeft->loadNode(nodeLeftFile);


cv::FileNode nodeRightFile=nodeRootFile["nodeRight"];
nodeRight=new NODE_EVALUATION(this,parentClassifier);
nodeRight->loadNode(nodeRightFile);

}
/*____________________________________________________________________*/


}


void NODE_EVALUATION::initializeFeatures()
{


if(!nodeIsTerminal)
{


//______________Se desaloja la pila, esto es por si los parámetros establecidos se quieren cambiar_______________//
if(!stackFeatures.empty()){
for(int i=0;i<stackFeatures.size();i++)
{

for(int j=0;j<stackFeatures[i].size();j++){
delete []stackFeatures[i][j];
}

}
stackFeatures.clear();
}



stackFeatures.reserve(parentClassifier->degrees.size());//Introducido el 5 de octubre del 2015

//________________________________________________________________________________________________________________//



int x1=feature->x%parentClassifier->widthImages;
int y1=feature->x/parentClassifier->widthImages;

int x2=feature->y%parentClassifier->widthImages;
int y2=feature->y/parentClassifier->widthImages;



for(int i=0;i<parentClassifier->degrees.size();i++){


/*____________MATRIZ DE ROTACIÓN__________________*/

/*

  |cos(alfa) -sin(alfa)|
R=|                    |
  |sin(alfa)  cos(alfa)|

*/

cv::Mat MR(2, 2,cv::DataType<double>::type);

if(flip && (i>=(parentClassifier->degrees.size()/2))){
MR.at<double>(0,0)=-std::cos(parentClassifier->degrees[i]*(M_PI/180));//cos(alfa)
MR.at<double>(0,1)=std::sin(parentClassifier->degrees[i]*(M_PI/180));//sin(alfa)
MR.at<double>(1,0)=std::sin(parentClassifier->degrees[i]*(M_PI/180));//-sin(alfa)
MR.at<double>(1,1)=std::cos(parentClassifier->degrees[i]*(M_PI/180));//cos(alfa)
}else{

MR.at<double>(0,0)=std::cos(parentClassifier->degrees[i]*(M_PI/180));//cos(alfa)
MR.at<double>(0,1)=std::sin(parentClassifier->degrees[i]*(M_PI/180));//sin(alfa)
MR.at<double>(1,0)=-std::sin(parentClassifier->degrees[i]*(M_PI/180));//-sin(alfa)
MR.at<double>(1,1)=std::cos(parentClassifier->degrees[i]*(M_PI/180));//cos(alfa)

}


/*________________________________________________*/


cv::Mat V1(2, 1,cv::DataType<double>::type);
V1.at<double>(0,0)=(x1-(parentClassifier->widthImages/2));
V1.at<double>(1,0)=((parentClassifier->highImages/2)-y1);

cv::Mat V2(2, 1,cv::DataType<double>::type);
V2.at<double>(0,0)=(x2-(parentClassifier->widthImages/2));
V2.at<double>(1,0)=((parentClassifier->highImages/2)-y2);

V1=MR.t()*V1;
V2=MR.t()*V2;
/*____________El 0.5 es por el error numérico________________*/
V1.at<double>(0,0)=(int)(V1.at<double>(0,0)+(parentClassifier->widthImages/2)+0.5);
V1.at<double>(1,0)=(int)((parentClassifier->highImages/2)-V1.at<double>(1,0)+0.5);
V2.at<double>(0,0)=(int)(V2.at<double>(0,0)+(parentClassifier->widthImages/2)+0.5);
V2.at<double>(1,0)=(int)((parentClassifier->highImages/2)-V2.at<double>(1,0)+0.5);
/*____________________________________________________________*/


std::vector<int *> stackTemp;
//___________________Introducido el 5 de octubre del 2015____________________________//
int count=0;
for(int sizeBase=parentClassifier->sizeBaseEvaluation;sizeBase<=parentClassifier->sizeMaxWindow;sizeBase=(parentClassifier->factorScaleWindow)*sizeBase)
{
count++;
}
//____________________________________________________________________________________//
stackTemp.reserve(count);

for(int sizeBase=parentClassifier->sizeBaseEvaluation;sizeBase<=parentClassifier->sizeMaxWindow;sizeBase=(parentClassifier->factorScaleWindow)*sizeBase)
{

int ky=(sizeBase/parentClassifier->highImages),kx=(sizeBase/parentClassifier->widthImages);

int *vecTemp=new int[4];
vecTemp[0]=ky*V1.at<double>(1,0);
vecTemp[1]=kx*V1.at<double>(0,0);
vecTemp[2]=ky*V2.at<double>(1,0);
vecTemp[3]=kx*V2.at<double>(0,0);

stackTemp.push_back(vecTemp);

}

stackFeatures.push_back(stackTemp);


}

nodeLeft->initializeFeatures();
nodeRight->initializeFeatures();
}



}





void NODE_EVALUATION::flipFeatures(bool flip)
{

this->flip=flip;

if(!nodeIsTerminal)
{

 nodeLeft->flipFeatures(flip);
 nodeRight->flipFeatures(flip);

}



}






double NODE_EVALUATION::evaluateNode(cv::Mat &image,int scale,int orderDegrees)
{
return (this->*pointerToFunctionEvaluation)(image,scale,orderDegrees);
}

double NODE_EVALUATION::evaluateNodeNoTerminal(cv::Mat &image,int scale,int orderDegrees)
{

int *vecTemp=(stackFeatures[orderDegrees])[scale];

int p1=image.at<uchar>(vecTemp[0],vecTemp[1]);
int p2=image.at<uchar>(vecTemp[2],vecTemp[3]);

/*
int delta=1;
int p1=(image.at<uchar>(vecTemp[0],vecTemp[1])+image.at<uchar>(vecTemp[0]+delta*scale,vecTemp[1])+image.at<uchar>(vecTemp[0],vecTemp[1]+delta*scale)+image.at<uchar>(vecTemp[0]-delta*scale,vecTemp[1])+image.at<uchar>(vecTemp[0],vecTemp[1]-delta*scale))/5;
int p2=(image.at<uchar>(vecTemp[2],vecTemp[3])+image.at<uchar>(vecTemp[2]+delta*scale,vecTemp[3])+image.at<uchar>(vecTemp[2],vecTemp[3]+delta*scale)+image.at<uchar>(vecTemp[2]-delta*scale,vecTemp[3])+image.at<uchar>(vecTemp[2],vecTemp[3]-delta*scale))/5;
*/

if(NPD_VALUES_FOR_EVALUATION[p1][p2]>threshold) return nodeRight->evaluateNode(image,scale,orderDegrees); /*Se envía a el nodo derecho*/
return nodeLeft->evaluateNode(image,scale,orderDegrees);/*Se envía a el nodo izquierdo*/

}


double NODE_EVALUATION::evaluationNodeTerminal(cv::Mat &image,int scale,int orderDegrees)
{
//std::cout<<"en nodo="<<yt<<"\n";
return yt;
}



TREE_TRAINING_EVALUATION::~TREE_TRAINING_EVALUATION()
{

delete nodeRoot;//Se elimina el nodo raíz que llamara a los destructores de sus hijos en secuencia
}


void TREE_TRAINING_EVALUATION::initializeFeatures()
{
nodeRoot->initializeFeatures();
}


void TREE_TRAINING_EVALUATION::flipFeatures(bool flip)
{

this->flip=flip;
nodeRoot->flipFeatures(flip);

}


void TREE_TRAINING_EVALUATION::loadWeakLearn(cv::FileNode weakLearnsTrees, int num)
{

std::string str_tree;
std::stringstream sstm;
sstm<<"tree_"<<num;
str_tree=sstm.str();


cv::FileNode tree=weakLearnsTrees[str_tree];
cv::FileNode information=tree["information"];

cv::FileNode nodeRootFile=tree["nodeRoot"];
nodeRoot=new NODE_EVALUATION(NULL,parentClassifier);
nodeRoot->loadNode(nodeRootFile);

}



double TREE_TRAINING_EVALUATION::evaluateTree(cv::Mat &image,int scale,int orderDegrees)
{

return nodeRoot->evaluateNode(image,scale,orderDegrees);
}



STRONG_LEARN_EVALUATION::~STRONG_LEARN_EVALUATION()
{


for(int i=0;i<weakLearns.size();i++)
delete weakLearns[i];
weakLearns.clear();

}


void STRONG_LEARN_EVALUATION::initializeFeatures()
{

for(int i=0;i<weakLearns.size();i++)
weakLearns[i]->initializeFeatures();

}

void STRONG_LEARN_EVALUATION::flipFeatures(bool flip)
{

this->flip=flip;

for(int i=0;i<weakLearns.size();i++)
weakLearns[i]->flipFeatures(flip);

}


void STRONG_LEARN_EVALUATION::loadStrongLearn(cv::FileNode fileStrongLearn, int stage)
{


std::string str_stage;
std::stringstream sstm;
sstm<<"stage_"<<stage;
str_stage=sstm.str();

cv::FileNode strongLearn=fileStrongLearn[str_stage];
cv::FileNode information=strongLearn["information"];
threshold=(double)information["threshold"];

/*A continuación se carga cada weakLearn o arbol*/
cv::FileNode weakLearnsTrees=strongLearn["weakLearns"];
weakLearns.reserve(weakLearnsTrees.size());

for(int i=0;i<weakLearnsTrees.size();i++)
{
TREE_TRAINING_EVALUATION *treeAux=new TREE_TRAINING_EVALUATION(parentClassifier);//Se enlaza con el clasificador padre
treeAux->loadWeakLearn(weakLearnsTrees,i);/*Aquí se carga cada clasificador*/
weakLearns.push_back(treeAux);
}




std::cout<<"stage "<<stage<<" load\n";



}

double STRONG_LEARN_EVALUATION::evaluateStrongLearnWithZeroThreshold(cv::Mat &image,int scale,int orderDegrees)
{

double evaluation=0;

for(int i=0;i<weakLearns.size();i++)
evaluation=evaluation+weakLearns[i]->evaluateTree(image,scale,orderDegrees);

return evaluation;
}


bool STRONG_LEARN_EVALUATION::evaluateStrongLearn(cv::Mat &image,int scale,int orderDegrees)
{


if(evaluateStrongLearnWithZeroThreshold(image,scale,orderDegrees)>=threshold)
return true;/*Se clasifica como positivo*/
else 
return false;/*Se clasifica como negativo*/

}


#if EVALUATION_FDDB==1

bool STRONG_LEARN_EVALUATION::FDDB_evaluateStrongLearn(cv::Mat &image,int scale,int orderDegrees)
{

scoreDetection=evaluateStrongLearnWithZeroThreshold(image,scale,orderDegrees);

if(scoreDetection>=threshold)
return true;/*Se clasifica como positivo*/
else 
return false;/*Se clasifica como negativo*/


}

#endif


CASCADE_CLASSIFIERS_EVALUATION::CASCADE_CLASSIFIERS_EVALUATION(std::string nameFile):NPD(NULL)
{

fileCascadeClassifier=new cv::FileStorage(nameFile,cv::FileStorage::READ);
loadCascadeClasifier();

//______________ESTO DEBE HACERSE DESPUÉS DE CARGAR EL CLASIFICADOR COMPLETO_____________________
flipFeatures(false);
degrees.push_back(0);//Cero grados
factorScaleWindow=1.2;//1.2 fue el valor que mejor funciono en las pruebas
stepWindow=0.2;//0.2 fue el valor que mejor funciono en las pruebas
sizeMaxWindow=640;//Generalmente las imágenes no sobrepasaran este tamaño, para tamaños mayores modifíquese
initializeFeatures();//Se inicializan las características con los parámetros anteriores
//_______________________________________________________________________________________________

//______Otros parámetros por defecto__________//
lineThicknessRectangles=1;
colorRectangles=cv::Scalar(0,255,0);//Por defecto el color es verde
groupThreshold=1;
eps=0.2;
flagActivateSkinColor=false;//Por defecto no se activa el skin color
hsvMin=cv::Scalar(0, 10, 60);//valor mínimo, funciona muy bien
hsvMax=cv::Scalar(20, 150, 255);//valor mínimo, funciona muy bien
flagExtractColorImages=false;//Por defecto las regiones detectadas se devuelven en escala de grises

numberClassifiersUsed=strongLearnsEvaluation.size();

//____________________________________________//

}

CASCADE_CLASSIFIERS_EVALUATION::~CASCADE_CLASSIFIERS_EVALUATION()
{

//Eliminamos la memoria dinámica del vector que almacena las NPD
if(NPD!=NULL){
delete []NPD;
NPD=NULL;
}

//Eliminamos la memoria dinámica de los strongLearns
for(int i=0;i<strongLearnsEvaluation.size();i++)
delete strongLearnsEvaluation[i];
strongLearnsEvaluation.clear();


}


void CASCADE_CLASSIFIERS_EVALUATION::initializeFeatures()
{

for(int i=0;i<strongLearnsEvaluation.size();i++)
strongLearnsEvaluation[i]->initializeFeatures();


zsBackground=0.8*double(sizeMaxWindow);
szImg=cv::Size(0,0);

}


void CASCADE_CLASSIFIERS_EVALUATION::flipFeatures(bool flip)
{

this->flip=flip;

for(int i=0;i<strongLearnsEvaluation.size();i++)
  strongLearnsEvaluation[i]->flipFeatures(flip);


}



void CASCADE_CLASSIFIERS_EVALUATION::setDegreesDetections(std::vector<double> myDegrees)
{
degrees=myDegrees;


int tmp_sz=degrees.size();
if(flip){
for(int i=0;i<tmp_sz;i++)
degrees.push_back(degrees[i]);
}


}

void CASCADE_CLASSIFIERS_EVALUATION::setSizeBase(int sizeBase)
{
sizeBaseEvaluation=sizeBase;
}

void CASCADE_CLASSIFIERS_EVALUATION::setFactorScaleWindow(double factorScale)
{
factorScaleWindow=factorScale;
}

void CASCADE_CLASSIFIERS_EVALUATION::setStepWindow(double factorStep)
{
stepWindow=factorStep;
}


void CASCADE_CLASSIFIERS_EVALUATION::setSizeMaxWindow(double maxSize)
{
sizeMaxWindow=maxSize;
}

void CASCADE_CLASSIFIERS_EVALUATION::setLineThicknessRectangles(int thicknessRectangles)
{

lineThicknessRectangles=thicknessRectangles;

}

void CASCADE_CLASSIFIERS_EVALUATION::setColorRectangles(const cv::Scalar& color)
{

colorRectangles=color;

}

void CASCADE_CLASSIFIERS_EVALUATION::setGroupThreshold(int threshold)
{
groupThreshold=threshold;
}


void CASCADE_CLASSIFIERS_EVALUATION::setEps(double myEps)
{
eps=myEps;
}

void CASCADE_CLASSIFIERS_EVALUATION::setFlagActivateSkinColor(bool activateSkinColor)
{
flagActivateSkinColor=activateSkinColor;
}

void CASCADE_CLASSIFIERS_EVALUATION::setFlagExtractColorImages(bool extractColorImages)
{
flagExtractColorImages=extractColorImages;
}

void CASCADE_CLASSIFIERS_EVALUATION::setNumberClassifiersUsed(int number)
{

if(number<1)
numberClassifiersUsed=1;
else if(number>strongLearnsEvaluation.size())
numberClassifiersUsed=strongLearnsEvaluation.size();
else
numberClassifiersUsed=number;

}

void CASCADE_CLASSIFIERS_EVALUATION::setHsvMin(const cv::Scalar& hsv)
{
hsvMin=hsv;
}

void CASCADE_CLASSIFIERS_EVALUATION::setHsvMax(const cv::Scalar& hsv)
{
hsvMax=hsv;
}

int CASCADE_CLASSIFIERS_EVALUATION::getNumberStrongLearns()const
{
return strongLearnsEvaluation.size();
}

double CASCADE_CLASSIFIERS_EVALUATION::getSizeMaxWindow()const
{
return sizeMaxWindow;
}

cv::Scalar CASCADE_CLASSIFIERS_EVALUATION::getHsvMin()const
{
return hsvMin;
}

cv::Scalar CASCADE_CLASSIFIERS_EVALUATION::getHsvMax()const
{
return hsvMax;
}

void CASCADE_CLASSIFIERS_EVALUATION::generateFeatures()
{

int p=widthImages*highImages;
int numberFeature=p*(p-1)/2;//Esto es producto de sumar (p-1)+(p-2)....+1

if(NPD!=NULL){
delete []NPD;/*Esto liberara la memoria si se produce una doble llamada, debido a que el alto y ancho base puede cambiar*/
NPD=NULL;
}

NPD=new cv::Point2i[numberFeature];//Pidiendo espacio para almacenar las características
//____llenando con los valores de cada coordenada___
int count=0;
for(int i=0;i<p;i++){
for(int j=i+1;j<p;j++){
NPD[count]=cv::Point2i(i,j);/*para el calculo aplíquese (xi-xj)/(xi+xj), donde xi y xj son los píxeles en las coordenadas i y j respectivamente.*/
count++;
 }
}

}




void CASCADE_CLASSIFIERS_EVALUATION::loadCascadeClasifier()
{

/*____________________Aquí empieza la lectura del clasificador______________________*/

cv::FileNode information=(*fileCascadeClassifier)["information"];

/*__________AQUÍ SE ACTUALIZA VARIABLES MUY IMPORTANTES_____________*/

widthImages=(int)information["G_WIDTH_IMAGE"];
highImages=(int)information["G_HEIGHT_IMAGE"];
sizeBaseEvaluation=std::min(widthImages,highImages);//Por defecto su tamaño es el de lado menor del tamaño de la imagen
generateFeatures();//Debe llamarse después de que se asigne valores a widthImages y highImages

/*_______________________________________________________________*/


/*A continuación se carga cada strong learn*/
cv::FileNode cascade_classifiers=(*fileCascadeClassifier)["cascade_classifiers"];
strongLearnsEvaluation.reserve(cascade_classifiers.size());

for(int i=0;i<cascade_classifiers.size();i++)
{
STRONG_LEARN_EVALUATION *strongLearnAux=new STRONG_LEARN_EVALUATION(this);
strongLearnAux->loadStrongLearn(cascade_classifiers,i);/*Aquí se carga cada clasificador*/
strongLearnsEvaluation.push_back(strongLearnAux);
}

fileCascadeClassifier->release();
if(fileCascadeClassifier!=NULL)delete fileCascadeClassifier;



}




bool CASCADE_CLASSIFIERS_EVALUATION::evaluateClassifier(cv::Mat &image,int scale,int orderDegrees)
{

double evaluation=0;

for(int i=0;i<numberClassifiersUsed;i++)
{

evaluation=evaluation+strongLearnsEvaluation[i]->evaluateStrongLearnWithZeroThreshold(image,scale,orderDegrees);
if(!(evaluation>(strongLearnsEvaluation[i]->threshold))) return false; /*Se clasifica como label negativo*/

}


return true;/*Se clasifica como label positivo*/

}


bool CASCADE_CLASSIFIERS_EVALUATION::evaluateClassifier(cv::Mat &image,int scale,int orderDegrees, int begin, int end, double &evaluation_ant)
{


for(int i=begin;i<end;i++)
{

evaluation_ant=evaluation_ant+strongLearnsEvaluation[i]->evaluateStrongLearnWithZeroThreshold(image,scale,orderDegrees);
if(!(evaluation_ant>(strongLearnsEvaluation[i]->threshold))) return false; /*Se clasifica como label negativo*/

}

return true;/*Se clasifica como label positivo*/


}


//__________________________________A CONTINUACIÓN SE DECLARAN DIFERENTES FUNCIONES DE DETECCIÓN____________________________________//

void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesUngrouped(cv::Mat &image)
{

if(szImg!=image.size()){
ImageBackground=cv::Mat::zeros(sizeMaxWindow+2*zsBackground,sizeMaxWindow+2*zsBackground,CV_8UC1);
imageGray=ImageBackground(cv::Range(zsBackground,zsBackground+image.rows),cv::Range(zsBackground,zsBackground+image.cols));
szImg=image.size();
}

cvtColor(image,imageGray,CV_BGR2GRAY);//Convirtiendo a escala de grises


if(flagActivateSkinColor){
cvtColor(image,hsv, CV_BGR2HSV);
inRange(hsv,hsvMin,hsvMax, bw);
integral(bw,integralBw,CV_32S);
}


int idx=0;

if(flagActivateSkinColor){
//______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));

bool flagDetected=false;
double myDegree=0;
int num=0;


double sum2=integralBw.at<int>(bottomRight_x+1,bottomRight_y+1)+integralBw.at<int>(topLeft_x,topLeft_y)-(integralBw.at<int>(bottomRight_x+1,topLeft_y)+integralBw.at<int>(topLeft_x,bottomRight_y+1));//Evaluación imagen integral

if(sum2>76.5*sizeBase*sizeBase){//76.5=255*0.3

for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees++){
if(evaluateClassifier(window,idx,orderDegrees)){

myDegree=myDegree+degrees[orderDegrees];
flagDetected=true;
num++;

}
}

}


if(flagDetected==true)
{

myDegree=myDegree/num;
cv::RotatedRect rectRotate(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-myDegree);
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);

}

 
}
}

}
//_______________________________________________________________________________________________________________________//
}else{
//_______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));

bool flagDetected=false;
double myDegree=0;
int num=0;

for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees++){
if(evaluateClassifier(window,idx,orderDegrees)){

myDegree=myDegree+degrees[orderDegrees];
flagDetected=true;
num++;

}
}



if(flagDetected==true)
{

myDegree=myDegree/num;
cv::RotatedRect rectRotate(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-myDegree);
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);

}

 
}
}

}
//___________________________________________________________________________________________________//
}








}



void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesUngrouped_2(cv::Mat &image)
{

if(szImg!=image.size()){
ImageBackground=cv::Mat::zeros(sizeMaxWindow+2*zsBackground,sizeMaxWindow+2*zsBackground,CV_8UC1);
imageGray=ImageBackground(cv::Range(zsBackground,zsBackground+image.rows),cv::Range(zsBackground,zsBackground+image.cols));
szImg=image.size();
}

cvtColor(image,imageGray,CV_BGR2GRAY);//Convirtiendo a escala de grises



int idx=0;


//______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));

bool flagDetected=false;
double myDegree=0;


std::vector<double> evaluation_degree_stage1;
evaluation_degree_stage1.reserve(3);
std::vector<double> evaluation_degree_stage2;
evaluation_degree_stage1.reserve(3);
std::vector<double> evaluation_degree_stage3;
evaluation_degree_stage1.reserve(3);
std::vector<double> evaluation_degree_stage4;
evaluation_degree_stage1.reserve(3);



for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees=orderDegrees+1){
double evaluation=0.0;
if(evaluateClassifier(window,idx,orderDegrees,0,16,evaluation)){

//flagDetected=true;
//evaluation_degree_stage1.push_back(evaluation);


//______________________________//


myDegree=degrees[orderDegrees];

cv::RotatedRect rectRotate(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-myDegree);
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);




//______________________________//


}else {
evaluation_degree_stage1.push_back(-std::numeric_limits<double>::infinity());
}

}




if(flagDetected)
{

//int dis=std::distance(evaluation_degree_stage1.begin(),std::max_element(evaluation_degree_stage1.begin(), evaluation_degree_stage1.end()));

//std::cout<<dis<<"\n";


/*
switch(dis)
{

case 0:
{
double evaluation=2+evaluation_degree_stage1[0];
for(int orderDegrees=0;orderDegrees<8;orderDegrees=orderDegrees+1){
if(evaluateClassifier(window,idx,orderDegrees,10,16,evaluation)){


//______________________________//

myDegree=degrees[orderDegrees];

cv::RotatedRect rectRotate(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-myDegree);
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);


//______________________________//



}
}


}
case 1:
{
double evaluation=2+evaluation_degree_stage1[1];
for(int orderDegrees=4;orderDegrees<14;orderDegrees=orderDegrees+1){
if(evaluateClassifier(window,idx,orderDegrees,10,16,evaluation)){

//______________________________//

myDegree=degrees[orderDegrees];

cv::RotatedRect rectRotate(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-myDegree);
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);


//______________________________//



}
}

}
case 2:
{
double evaluation=2+evaluation_degree_stage1[2];
for(int orderDegrees=10;orderDegrees<degrees.size();orderDegrees=orderDegrees+1){
if(evaluateClassifier(window,idx,orderDegrees,10,16,evaluation)){

//______________________________//

myDegree=degrees[orderDegrees];

cv::RotatedRect rectRotate(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-myDegree);
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);


//______________________________//



}
}

}

}

*/

}



/*
if(flagDetected)
{
  std::cout<<"\n";
  for(int zz=0;zz<3;zz++)
  std::cout<<(evaluation_degree_stage1[zz])<<" ";
  std::cout<<"\n";
}
*/






}
}
}





}

//___________________________________________________________________________________________________//













void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesGroupedZeroDegrees(cv::Mat &image,std::vector<cv::Mat> *listDetectedObjects,std::vector<cv::Rect> *coordinatesDetectedObjects,bool doubleDetectedList,bool paintDetections)
{



if(szImg!=image.size()){
ImageBackground=cv::Mat::zeros(sizeMaxWindow+2*zsBackground,sizeMaxWindow+2*zsBackground,CV_8UC1);
imageGray=ImageBackground(cv::Range(zsBackground,zsBackground+image.rows),cv::Range(zsBackground,zsBackground+image.cols));
szImg=image.size();
}

cvtColor(image,imageGray,CV_BGR2GRAY);//Convirtiendo a escala de grises


if(flagActivateSkinColor){
cvtColor(image,hsv, CV_BGR2HSV);
inRange(hsv,hsvMin,hsvMax, bw);
integral(bw,integralBw,CV_32S);
}


int idx=0;

if(flagActivateSkinColor){
//_______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));

//___________________________________________________________________________________________//
double sum2=integralBw.at<int>(bottomRight_x+1,bottomRight_y+1)+integralBw.at<int>(topLeft_x,topLeft_y)-(integralBw.at<int>(bottomRight_x+1,topLeft_y)+integralBw.at<int>(topLeft_x,bottomRight_y+1));//Evaluación imagen integral

if(sum2>76.5*sizeBase*sizeBase){//76.5=255*0.3

for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees++){
if(evaluateClassifier(window,idx,orderDegrees)){

cv::Rect rectTemp(j,i,sizeBase,sizeBase);
windowsCandidates.push_back(rectTemp);

}
}

}
//___________________________________________________________________________________________//

 
}
}

}
//_____________________________________________________________________________________________________________________//

}else{

//_______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));

//___________________________________________________________________________________________//


for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees++){
if(evaluateClassifier(window,idx,orderDegrees)){

cv::Rect rectTemp(j,i,sizeBase,sizeBase);
windowsCandidates.push_back(rectTemp);

}
}
//___________________________________________________________________________________________//


  
}
}

}
//_____________________________________________________________________________________________________________________//
}





if(doubleDetectedList==true){
/*Esto se hace debido a que la función cv::groupRectangles eliminara el grupo que solo posea un rectángulo, así que doblamos la lista para que estos no sean eliminados debido a que posiblemente sean escasos cuando la taza de falsos negativos es muy baja*/
//________________________________________________________________//
int sz=windowsCandidates.size();
for(int i=0;i<sz;i++)
windowsCandidates.push_back(cv::Rect(windowsCandidates[i]));
//________________________________________________________________//
}

//Agrupamos los rectángulos similares
cv::groupRectangles(windowsCandidates,groupThreshold,eps);



/*NOTA: Extraemos primero los rectángulos debido a que si lo hacemos después de pintar los rectángulos la imagen extraída también queda coloreada con las lineas que dibujan rectángulo*/
//_________Aquí extraemos las imágenes detectadas_________________
if(listDetectedObjects!=NULL)
{

if(flagExtractColorImages==true)
{

for(int i=0;i<windowsCandidates.size();i++)
listDetectedObjects->push_back(image(windowsCandidates[i]).clone());

}else
{

for(int i=0;i<windowsCandidates.size();i++)
listDetectedObjects->push_back(imageGray(windowsCandidates[i]).clone());

}

}
//_________________________________________________________________


if(paintDetections){
//Pintamos los rectángulos
for(int i=0;i<windowsCandidates.size();i++)
cv::rectangle(image,windowsCandidates[i],colorRectangles,lineThicknessRectangles);
}

if(coordinatesDetectedObjects!=NULL)
(*coordinatesDetectedObjects)=windowsCandidates;/*Coordenadas de los rectángulos detectados con respecto a las coordenadas de la imagen de entrada*/

windowsCandidates.clear();

}







//______________________________________________________________________________________________________________//


//La clase a continuación es una modificación de la clase SimilarRects de openCV
class SimilarRectsRotated
{
public:
    SimilarRectsRotated(double _eps) : eps(_eps) {}
    inline bool operator()(const cv::RotatedRect& r1, const cv::RotatedRect& r2) const
    {
        double delta = eps*(std::min(r1.size.width, r2.size.width) + std::min(r1.size.height, r2.size.height))*0.5;
        


           return std::abs(r1.center.x - r2.center.x) <= delta &&
            std::abs(r1.center.y - r2.center.y) <= delta &&
            std::abs(r1.size.width - r2.size.width) <= delta &&
            std::abs(r1.size.height-r2.size.height) <= delta;
      
    }
    double eps;
};



//La función a continuación es una modificación de la función groupRectangle de openCV
void groupRectanglesRotated(std::vector<cv::RotatedRect>& rectList, int groupThreshold, double eps)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        return;
    }

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRectsRotated(eps));

    std::vector<cv::RotatedRect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].center.x += rectList[i].center.x;
        rrects[cls].center.y += rectList[i].center.y;
        rrects[cls].size.width += rectList[i].size.width;
        rrects[cls].size.height += rectList[i].size.height;
        rrects[cls].angle += rectList[i].angle;
        rweights[cls]++;
    }


    for( i = 0; i < nclasses; i++ )
    {
        cv::RotatedRect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] =cv::RotatedRect(cv::Point2f(s*rrects[i].center.x,s*rrects[i].center.y),cv::Size2f(s*rrects[i].size.width,s*rrects[i].size.height),s*rrects[i].angle);


  
    }

    rectList.clear();


    for( i = 0; i < nclasses; i++ )
    {
        cv::RotatedRect r1 = rrects[i];
        int n1 = rweights[i];
        // filter out rectangles which don't have enough similar rectangles
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            cv::RotatedRect r2 = rrects[j];

            int dx = cv::saturate_cast<int>( r2.size.width * eps );
            int dy = cv::saturate_cast<int>( r2.size.height * eps );

            if( i != j &&
                r1.center.x >= r2.center.x - dx &&
                r1.center.y >= r2.center.y - dy &&
                r1.center.x + r1.size.width <= r2.center.x + r2.size.width + dx &&
                r1.center.y + r1.size.height <= r2.center.y + r2.size.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
        }
    }
}





void CASCADE_CLASSIFIERS_EVALUATION::detectObjectRectanglesRotatedGrouped(cv::Mat &image,std::vector<cv::Mat> *listDetectedObjects,std::vector<cv::RotatedRect> *coordinatesDetectedObjects,bool paintDetections)
{


cvtColor(image,imageGray,CV_BGR2GRAY);//Convirtiendo a escala de grises




int idx=0;


//__________________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));


//______________________________________________________________________________//




double evaluation_ant_s1=0,evaluation_ant_s2=0;
evaluateClassifier(window,idx,0,0,3,evaluation_ant_s1);
evaluateClassifier(window,idx,1,0,3,evaluation_ant_s2);

if(evaluation_ant_s1>evaluation_ant_s2){


 if(evaluateClassifier(window,idx,0,3,15,evaluation_ant_s1)){

  cv::RotatedRect windowTemp(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-degrees[0]);
  windowsCandidatesRotated.push_back(windowTemp);
  //std::cout<<evaluation_ant_s1-evaluation_ant_s2<<" <---------\n";

 }
 
}else{

 if(evaluateClassifier(window,idx,1,3,15,evaluation_ant_s2)){

  cv::RotatedRect windowTemp(cv::Point2f(j+(sizeBase/2),i+(sizeBase/2)),cv::Size2f(sizeBase,sizeBase),-degrees[1]);
  windowsCandidatesRotated.push_back(windowTemp);
  //std::cout<<evaluation_ant_s2-evaluation_ant_s1<<" --------->\n";
 
 }

}
//______________________________________________________________________________//






  
}
}

}
//_____________________________________________________________________________________________________________________//





groupRectanglesRotated(windowsCandidatesRotated,groupThreshold,eps);





//Pintamos los rectángulos
for(int i=0;i<windowsCandidatesRotated.size();i++)
{

cv::RotatedRect rectRotate=windowsCandidatesRotated[i];
cv::Point2f vertices[4];
rectRotate.points(vertices);
for (int i = 0; i < 4; i++)
cv::line(image, vertices[i], vertices[(i+1)%4],colorRectangles,lineThicknessRectangles);

}





windowsCandidatesRotated.clear();



}




//LOS MÉTODOS CLASES Y FUNCIONES A CONTINUACÍON DECLARADOS SOLO SON ÚTILES PARA LA EVALUACIÓN DE LA BASE DE DATOS FDDB
#if EVALUATION_FDDB==1


bool CASCADE_CLASSIFIERS_EVALUATION::FDDB_evaluateClassifier(cv::Mat &image,int scale,int orderDegrees)
{

for(int i=0;i<numberClassifiersUsed;i++)
if(!strongLearnsEvaluation[i]->FDDB_evaluateStrongLearn(image,scale,orderDegrees)) return false;/*Se clasifica como label negativo*/

scoreDetection=strongLearnsEvaluation[numberClassifiersUsed-1]->scoreDetection;//Se toma el score de la detección

return true;/*Se clasifica como label positivo*/

}



//Esta clase almacenara el rectángulo de la detección con su respectivo score 
class FDDB_RECT_AND_SCORES:public cv::Rect
{
public:
FDDB_RECT_AND_SCORES(){};
FDDB_RECT_AND_SCORES(int x,int y,int width,int height,double score=0):cv::Rect(x,y,width,height),score(score){}

FDDB_RECT_AND_SCORES(const FDDB_RECT_AND_SCORES &rect)
{
x=rect.x;
y=rect.y;
width=rect.width;
height=rect.height;
score=rect.score;
}

cv::Rect getRect()const
{
return cv::Rect(x,y,width,height);
}

double score;
};


//Esta clase es una modificación de la clase SimilarRects de openCV, sirve para saber si dos rectángulos son vecinos
class FDDB_SimilarRects
{
public:
    FDDB_SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const FDDB_RECT_AND_SCORES& r1, const FDDB_RECT_AND_SCORES& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

//Esta función es una modificación de la función groupRectangles de openCV, sirve para agrupar rectángulos de detecciones vecinas
void FDDB_groupRectangles(std::vector<FDDB_RECT_AND_SCORES>& rectList, int groupThreshold, double eps,
                     std::vector<int>* weights=NULL, std::vector<double>* levelWeights=NULL)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, FDDB_SimilarRects(eps));

    std::vector<FDDB_RECT_AND_SCORES> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    std::vector<int> rejectLevels(nclasses, 0);
    std::vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rrects[cls].score += rectList[i].score;
        rweights[cls]++;
    }

    bool useDefaultWeights = false;

    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }
    else
        useDefaultWeights = true;

    for( i = 0; i < nclasses; i++ )
    {
        FDDB_RECT_AND_SCORES r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = FDDB_RECT_AND_SCORES(cv::saturate_cast<int>(r.x*s),
        cv::saturate_cast<int>(r.y*s),
        cv::saturate_cast<int>(r.width*s),
        cv::saturate_cast<int>(r.height*s),
        s*r.score);
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();

    for( i = 0; i < nclasses; i++ )
    {
        FDDB_RECT_AND_SCORES r1 = rrects[i];
        int n1 = rweights[i];
        double w1 = rejectWeights[i];
        int l1 = rejectLevels[i];

        // filter out rectangles which don't have enough similar rectangles
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            FDDB_RECT_AND_SCORES r2 = rrects[j];

            int dx = cv::saturate_cast<int>( r2.width * eps );
            int dy = cv::saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(useDefaultWeights ? n1 : l1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}


/*Esta función detecta los rostros en una imagen y devuelve los rectángulos de dichas coordenadas y los scores correspondientes a cada detección que el software dado en http://vis-www.cs.umass.edu/fddb/ requiere*/ 
void CASCADE_CLASSIFIERS_EVALUATION::FDDB_detectObjectRectanglesGroupedZeroDegrees(cv::Mat &image,std::vector<cv::Rect>& rectanglesDetected,std::vector<double>& scores,bool doubleDetectedList)
{

std::vector<FDDB_RECT_AND_SCORES> fddbWindowsCandidates;


if(szImg!=image.size()){
ImageBackground=cv::Mat::zeros(sizeMaxWindow+2*zsBackground,sizeMaxWindow+2*zsBackground,CV_8UC1);
imageGray=ImageBackground(cv::Range(zsBackground,zsBackground+image.rows),cv::Range(zsBackground,zsBackground+image.cols));
szImg=image.size();
}

cvtColor(image,imageGray,CV_BGR2GRAY);//Convirtiendo a escala de grises


if(flagActivateSkinColor){
cvtColor(image,hsv, CV_BGR2HSV);
inRange(hsv,hsvMin,hsvMax, bw);
integral(bw,integralBw,CV_32S);
}

int idx=0;

if(flagActivateSkinColor){

//_______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));

//___________________________________________________________________________________________//
double sum2=integralBw.at<int>(bottomRight_x+1,bottomRight_y+1)+integralBw.at<int>(topLeft_x,topLeft_y)-(integralBw.at<int>(bottomRight_x+1,topLeft_y)+integralBw.at<int>(topLeft_x,bottomRight_y+1));//Evaluación imagen integral

if(sum2>76.5*sizeBase*sizeBase){//76.5=255*0.3

for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees++){
if(FDDB_evaluateClassifier(window,idx,orderDegrees)){

FDDB_RECT_AND_SCORES rectTemp(j,i,sizeBase,sizeBase,scoreDetection);
fddbWindowsCandidates.push_back(rectTemp);

}
}

}
//___________________________________________________________________________________________//



  
}
}

}
//_______________________________________________________________________________________________________________________//


}else{

//_______________________________________________________________________________________________________________________//
for(int sizeBase=sizeBaseEvaluation;sizeBase<=std::min(image.rows,image.cols);sizeBase=factorScaleWindow*sizeBase,idx++){
int step_x=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en filas*/
int step_y=sizeBase*stepWindow;/*En este factor la ventana de búsqueda avanzara en columnas*/
for(int i=0;i<=image.rows-sizeBase;i=i+step_x){
for(int j=0;j<=image.cols-sizeBase;j=j+step_y){

/*__________CALCULO DE LA POSICIÓN DE LA VENTANA______________*/
int topLeft_x=i;
int topLeft_y=j;
int bottomRight_x=sizeBase+i-1;
int bottomRight_y=sizeBase+j-1;
/*____________________________________________________________*/

cv::Mat window;
window=imageGray(cv::Range(topLeft_x,bottomRight_x+1),cv::Range(topLeft_y,bottomRight_y+1));


//___________________________________________________________________________________________//

for(int orderDegrees=0;orderDegrees<degrees.size();orderDegrees++){
if(FDDB_evaluateClassifier(window,idx,orderDegrees)){

FDDB_RECT_AND_SCORES rectTemp(j,i,sizeBase,sizeBase,scoreDetection);
fddbWindowsCandidates.push_back(rectTemp);

}
}
//___________________________________________________________________________________________//


  
}
}

}
//_______________________________________________________________________________________________________________________//

}


if(doubleDetectedList==true){
/*Esto se hace debido a que la función cv::groupRectangles eliminara el grupo que solo posea un rectángulo, así que doblamos la lista para que estos no sean eliminados debido a que posiblemente sean escasos cuando la taza de falsos negativos es muy baja*/
//________________________________________________________________//
int sz=fddbWindowsCandidates.size();
for(int i=0;i<sz;i++)
fddbWindowsCandidates.push_back(FDDB_RECT_AND_SCORES(fddbWindowsCandidates[i]));
//________________________________________________________________//
}

//Agrupamos los rectángulos similares
FDDB_groupRectangles(fddbWindowsCandidates,groupThreshold,eps);




for(int i=0;i<fddbWindowsCandidates.size();i++)
{
rectanglesDetected.push_back(fddbWindowsCandidates[i].getRect());
scores.push_back(fddbWindowsCandidates[i].score);
}


//Pintamos los rectángulos
for(int i=0;i<fddbWindowsCandidates.size();i++)
cv::rectangle(image,fddbWindowsCandidates[i].getRect(),colorRectangles,lineThicknessRectangles);



}




#endif
