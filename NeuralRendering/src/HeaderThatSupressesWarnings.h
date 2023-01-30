//Here because SDL and especially LibTorch have an excessive amount of warnings in their headers.
//CUDA also has 1, but that's mot that bad.
#pragma warning( push, 0 )
//just write 1 instead of 0 above and see for yourself how many warning there are.
//Also for some reasen the macros in this file do not contain all warnings. what a lie.
//#include <CodeAnalysis/Warnings.h>
//#pragma warning(disable: ALL_CODE_ANALYSIS_WARNINGS)
