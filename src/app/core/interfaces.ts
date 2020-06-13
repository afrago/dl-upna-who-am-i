//-------------------------------------------------------------
// defines 'Person Representation' interface to store training dataset
//-------------------------------------------------------------
export interface IPersonRep 
{
  ImageSrc: string; // this is to get the correct file
  ImageName: string;
  Class: number; 
  Label: string;
};
//-------------------------------------------------------------
// defines 'TrainingMetrics' interface to store training metrics
//-------------------------------------------------------------
export interface ITrainingMetrics 
{
  acc: number; // training accuracy value 
  ce: number; // cross entropy value 
  loss: number; // loss function value 
};




