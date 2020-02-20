Experiment 1:
	Regressor:      SCUT-FBP  1-400
	Discriminator:  SCUT-FBP  1-400
	Cycle:		SCUT-FBP  1-400
	Generator:	SCUT-FBP-V2  1-2000
	Test:		SCUT-FBP  400-500


Experiment 2:
	Regressor:      SCUT-FBP-V2  1-2000 
	Discriminator:  AFAD  1-4000
	Cycle:		SCUT-FBP-V2  1-2000
	Generator:	AFAD  4000-24000
	Test:           AFAD


Experiment 3:
	Regressor:      SCUT-FBP-V2  1-1800 
	Discriminator:  SCUT-FBP-V2  1-1800
	Cycle:		SCUT-FBP-V2  1-1800
	Generator:	SCUT-FBP-V2  1-1800
	Test:		SCUT-FBP-V2  1800-2000


Experiment 4:
	Regressor:      SCUT-FBP-V2  1-2000 
	Discriminator:  CelebA
	Cycle:		SCUT-FBP-V2  1-2000 
	Generator:	CelebA
	Test:		CelebA
