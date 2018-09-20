# Kohonen's Self-Organizing Map implementation

### Usage:
```
 $ python kohonen_som.py [-h|--help] [-d|--defaults] [-f|--file ...] -s|--shape ... -l|--lr ... [-r|--rseed ...] [-i|--iterations ...]
```
* -h or --help --> __Optional__ Display help information
* -d or --defaults --> __Optional__ Use default values for unspecified arguments
* -s or --shape --> __Required__ Specify the dimension of the output, comma-separated without space (_Ex: 50,50_)
* -l or --lr --> __Required__ Initial learning rate
* -r or --rseed --> __Optional__ Random seed value
* -i or --iterations --> __Optional__ Maximum number of iterations
* -f or --file --> __Optional__ CSV data file to use for training

Note:
* This code is written in Python3

###### References:
* __Section 9.3__ from _Neural Networks and Learning Machines by Simon Haykin, 3rd Ed, Prentice Hall_
* http://www.ai-junkie.com/ann/som/som1.html
* https://www.youtube.com/watch?v=wRcnNyoXm_M&list=PLQTSaPQDJvULaCV878HL37eNYo-YC1q8u&index=2
