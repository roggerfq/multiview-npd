#ifndef DETECTOR_H
#define DETECTOR_H


//stl
#include <iostream>
#include<vector>

//________________LIBRERIAS OPENCV___________________
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//______________________________________________________________________________________


/*
Si la siguiente bandera tiene el valor de 1, las funciones y clases relacionadas con la evaluación de un detector de rostro a cargo del 
software dado en http://vis-www.cs.umass.edu/fddb/ serán compiladas.
*/
#define EVALUATION_FDDB 1  


/*Predeclaración*/
class NODE_EVALUATION;
class TREE_TRAINING_EVALUATION;
class STRONG_LEARN_EVALUATION;
class CASCADE_CLASSIFIERS_EVALUATION;



typedef double(NODE_EVALUATION::*PointerToEvaluationNode_Evaluation)(cv::Mat &,int scale,int orderDegrees) ;
class NODE_EVALUATION
{

CASCADE_CLASSIFIERS_EVALUATION *parentClassifier;//A través de este puntero se tiene acceso a algunos datos

class NODE_EVALUATION *parent;//Apunta al nodo padre, si es el nodo inicial su valor debe ser cero
class NODE_EVALUATION *nodeLeft;//Apunta al nodo izquierdo, si no tiene un nodo que se derive de el debe apuntar a cero
class NODE_EVALUATION *nodeRight;//Apunta al nodo derecho, si no tiene un nodo que se derive de el debe apuntar a cero


bool nodeIsTerminal;
double threshold;
int numFeature;
cv::Point2i *feature;
std::vector< std::vector<int *> > stackFeatures;
double yt;

PointerToEvaluationNode_Evaluation pointerToFunctionEvaluation;/*Puntero a la función de evaluación*/

void setNodeAsTerminal();
public:
/*NOTA:EL parámetro NULL en parentClassifier es debido a que en el constructor de TREE_TRAINING_EVALUATION se crea inicialmente un puntero
de esta clase*/
NODE_EVALUATION(NODE_EVALUATION *myParent=NULL,CASCADE_CLASSIFIERS_EVALUATION *parentClassifier=NULL):parent(myParent),parentClassifier(parentClassifier),nodeLeft(NULL),nodeRight(NULL),feature(NULL),nodeIsTerminal(false){pointerToFunctionEvaluation=&NODE_EVALUATION::evaluateNodeNoTerminal;}
~NODE_EVALUATION();
void loadNode(cv::FileNode nodeRootFile);
void initializeFeatures();
void flipFeatures(bool flip);
double evaluateNode(cv::Mat &image,int scale,int orderDegrees);
double evaluateNodeNoTerminal(cv::Mat &image,int scale,int orderDegrees);/*Función de evaluación para nodo NODO terminal*/
double evaluationNodeTerminal(cv::Mat &image,int scale,int orderDegrees);/*Función de evaluación para nodo terminal*/


bool flip;
};

class TREE_TRAINING_EVALUATION
{

CASCADE_CLASSIFIERS_EVALUATION *parentClassifier;//A través de este puntero se tiene acceso a algunos datos

NODE_EVALUATION *nodeRoot;

public:
TREE_TRAINING_EVALUATION(CASCADE_CLASSIFIERS_EVALUATION *parentClassifier):nodeRoot(NULL),parentClassifier(parentClassifier){}
~TREE_TRAINING_EVALUATION();
void loadWeakLearn(cv::FileNode weakLearnsTrees, int num);
void initializeFeatures();
void flipFeatures(bool flip);
double evaluateTree(cv::Mat &image,int scale,int orderDegrees);

bool flip;

};



class STRONG_LEARN_EVALUATION
{

CASCADE_CLASSIFIERS_EVALUATION *parentClassifier;//A través de este puntero se tiene acceso a algunos datos

std::vector< TREE_TRAINING_EVALUATION *> weakLearns;
public:
STRONG_LEARN_EVALUATION(CASCADE_CLASSIFIERS_EVALUATION *parentClassifier):parentClassifier(parentClassifier){};
~STRONG_LEARN_EVALUATION();
void loadStrongLearn(cv::FileNode fileStrongLearn, int stage);
double evaluateStrongLearnWithZeroThreshold(cv::Mat &image,int scale,int orderDegrees);
bool evaluateStrongLearn(cv::Mat &image,int scale,int orderDegrees);/*Evaluación usando threshold*/
void initializeFeatures();
void flipFeatures(bool flip);
double threshold;
bool flip;

#if EVALUATION_FDDB==1
double scoreDetection;//scoreDetection es la confianza de la detección
bool FDDB_evaluateStrongLearn(cv::Mat &image,int scale,int orderDegrees);
#endif

};


class CASCADE_CLASSIFIERS_EVALUATION
{
std::vector< STRONG_LEARN_EVALUATION *> strongLearnsEvaluation;/*Almacena cada uno de los clasificadores fuertes en la cascada*/
int numberClassifiersUsed;/*Número de clasificadores a usar, su valor debe variar entre 1 y strongLearnsEvaluation.size(), por defecto su valor es strongLearnsEvaluation.size()*/
int widthImages;
int highImages;
cv::Point2i *NPD;
std::vector<double> degrees;//Estos son los grados de detección, por defecto 0 grados
int sizeBaseEvaluation;//Por defecto su tamaño es el de lado menor del tamaño de la imagen
double factorScaleWindow;//Factor por el cual la ventana se ensanchara
double stepWindow;//Factor por el cual la ventana avanzara según su tamaño
double sizeMaxWindow;//Tamaño máximo de la ventana de búsqueda


//Parámetros de el rectángulo que muestra la detección
int lineThicknessRectangles;//Grosor de las lineas que dibujan los rectángulos (por defecto es 1)
cv::Scalar colorRectangles;//Por defecto el color es verde
int groupThreshold;//Umbral mínimo a partir un grupo de rectángulos similares no se elimina
double eps;//Relativa diferencia mínima de los lados de dos un rectángulo a partir de los cuales se agrupan 
bool flagActivateSkinColor;/*Si esta variable es true se activa un sencillo algoritmo de skin color, por defecto es false,
tenga en cuenta que si va a detectar otro objeto que no sea una cara humana, o si el detector se va a enfrentar a condiciones lumínicas muy variantes, o si la imagen de entrada esta en escala de grises la detección se vera afectada por el skin color ya que se eliminaran zonas que no estén dentro rango de colores de la piel humana*/  
bool flagExtractColorImages;/*Controla si las imágenes detectadas son devueltas en color (de la imagen de entrada) o en escala de grises, por defecto es false, es decir las imagenes se devuelven en escala de grises*/

bool flip;


/*Variables importantes durante la ejecución*/
std::vector<cv::RotatedRect> windowsCandidatesRotated;
std::vector<cv::Rect> windowsCandidates;
cv::Mat ImageBackground;//Imagen fondo extra para evitar error al evaluar características que se salgan de los bordes
int zsBackground;//Ancho que debe tener la imagen a analizar para evitar evaluar características en coordenadas inexistentes
cv::Size szImg;//Ancho de la ultima imagen analizada
cv::Mat imageAndBackgroundColor;//Solo útil para la función detectObjectRectanglesRotatedGrouped
cv::Mat imageAndBackgroundGray;//Solo útil para la función detectObjectRectanglesRotatedGrouped
cv::Mat imageGray;/*Esta matriz sera usada por cada función de detección de alto nivel (incluidas las perteneciente a la evaluación de la base de datos FDDB) para incrustar la imagen de entrada que es mucho mas pequeña (ver variable zsBackground) y así evitar errores de acceso a memoria inexistente al analizar rectángulos (ventana deslizante) en los limites de la imagen de entrada*/ 
cv::Mat hsv;//Skin Color
cv::Mat bw;//Skin Color
cv::Mat integralBw;//Imagen integral de bw
cv::Scalar hsvMin;//Valor mínimo en el espacio hsv aceptado como skin color
cv::Scalar hsvMax;//Valor máximo en el espacio hsv aceptado como skin color

cv::FileStorage *fileCascadeClassifier;
void generateFeatures();
void loadCascadeClasifier();
public:
CASCADE_CLASSIFIERS_EVALUATION(std::string nameFile);
~CASCADE_CLASSIFIERS_EVALUATION();

void initializeFeatures();
void flipFeatures(bool flip=false);
//___Si se llama a alguna de las siguientes funciones, se debe llamar a initializeFeatures() para que el cambio tenga efecto____
void setDegreesDetections(std::vector<double> myDegrees);//Establece los grados de búsqueda de la ventana
void setSizeBase(int sizeBase);//Establece el tamaño menor que tomara la ventana de búsqueda
void setFactorScaleWindow(double factorScale);//Establece el factor en el cual la ventana de búsqueda se ensanchara
void setStepWindow(double factorStep);//Establece el factor por el cual la ventana avanzara según su tamaño
void setSizeMaxWindow(double maxSize);//Establece el tamaño máximo de la ventana de búsqueda
//______________________________________________________________________________________________________________________________
void setLineThicknessRectangles(int thicknessRectangles);
void setColorRectangles(const cv::Scalar& color);

/*
En general, debe tenerse en cuenta al momento de establecer los parámetros groupThreshold y eps lo siguiente:
1-Si el detector presenta pocas detecciones vecinas es decir , las detecciones adyacente sobre el mismo objeto son pocas, deberá establecerse 
groupThreshold a un valor pequeño para que de esta forma las detecciones que solo presenten un rectángulo no sean eliminadas, el valor mínimo que puede tomar groupThreshold es 1, debido a esto en algunas de las funciones de detección se dobla la lista internamente.
2-El parámetro eps, en general, funciona bien para valores entre 0.2 y 0.5, este controla la diferencia relativa de los lados de dos rectángulos a partir de la cual se declaran como vecino
*/
void setGroupThreshold(int threshold);
void setEps(double myEps);

void setFlagActivateSkinColor(bool activateSkinColor);
void setFlagExtractColorImages(bool extractColorImages);
void setNumberClassifiersUsed(int number);//Establece el número de clasificadores a usar
void setHsvMin(const cv::Scalar& hsv);
void setHsvMax(const cv::Scalar& hsv);

//________Funciones get_______________//
int getNumberStrongLearns()const;//Retorna el total de clasificadores fuertes de la cascada
double getSizeMaxWindow()const;//Retorna el tamaño máximo de la ventana de búsqueda
cv::Scalar getHsvMin()const;
cv::Scalar getHsvMax()const;


//____________________________________//

//Las funciones a continuación son las funciones de detección de mas bajo nivel posible
bool evaluateClassifier(cv::Mat &image,int scale,int orderDegrees);
bool evaluateClassifier(cv::Mat &image,int scale,int orderDegrees, int begin, int end, double &evaluation_ant);


//______________A CONTINUACIÓN SE DECLARAN ALGUNAS FUNCIONES DE DETECCIÓN ÚTILES_________________________________//
/*La función a continuación detecta un objeto (tiene opción de skin color) pero no agrupa los rectángulos similares*/
void detectObjectRectanglesUngrouped(cv::Mat &image);
void detectObjectRectanglesUngrouped_2(cv::Mat &image);
/*La función a continuación tiene la opción de doblar el número de detecciones en una lista interna antes de agrupar los rectángulos, esto es debido a que el algoritmo de openCV  cv::groupRectangles elimina los rectángulos cuyo grupo sea conformado por un número menor o igual a groupThreshold, esto es un problema cuando el clasificador tiene una taza de falsos positivos baja, pues las detecciones son muy localizadas y tienden a haber pocos vecinos, este problema también puede presentarse cuando la detección se realiza a un solo ángulo, así que doblando la lista se evita que muchas detecciones que son correctas sean eliminadas por el algoritmo*/
void detectObjectRectanglesGroupedZeroDegrees(cv::Mat &image,std::vector<cv::Mat> *listDetectedObjects=NULL,std::vector<cv::Rect> *coordinatesDetectedObjects=NULL,bool doubleDetectedList=true,bool paintDetections=true);
/*La función a continuación detecta el objeto en los ángulos establecidos y puede opcionalmente extraer dichas zonas de la imagen, normalizarlas al ángulo estándar de entrenamiento y devolverlas en la lista std::vector<cv::Mat> listDetectedObjects */
void detectObjectRectanglesRotatedGrouped(cv::Mat &image,std::vector<cv::Mat> *listDetectedObjects=NULL,std::vector<cv::RotatedRect> *coordinatesDetectedObjects=NULL,bool paintDetections=true);
//_______________________________________________________________________________________________________________//


/*Los siguientes métodos y variables siguientes solo son útiles para mi tesis de pregrado, dichos métodos están encargados de extraer los rectángulos con los score de cada detección y sera usada por el software dado por la base de datos FDDB (http://vis-www.cs.umass.edu/fddb/), para claridad cada función o clase relacionada con la siguientes funciones y variables iniciaran su nombre con las siglas FDDB*/
#if EVALUATION_FDDB==1
//La función a continuación se usa para la evaluación de la base de datos FDDB
double scoreDetection;//scoreDetection es la confianza de la detección
bool FDDB_evaluateClassifier(cv::Mat &image,int scale,int orderDegrees);
void FDDB_detectObjectRectanglesGroupedZeroDegrees(cv::Mat &image,std::vector<cv::Rect>& rectanglesDetected,std::vector<double>& scores,bool doubleDetectedList=true);
#endif

friend class STRONG_LEARN_EVALUATION;
friend class TREE_TRAINING_EVALUATION;
friend class NODE_EVALUATION;

};




#endif
