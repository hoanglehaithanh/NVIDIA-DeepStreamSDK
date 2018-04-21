SHELL=/bin/sh

CUDAVERSION?=cuda-9.0
ifdef VERBOSE
AT=
else
AT=@
endif

samples=decPerf nvDecInfer_classification nvDecInfer_detection

.PHONY: all clean help
all: 
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample);)

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample);)

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)";)
	$(AT)echo "Commands:"
	$(AT)echo "all - build all samples."
	$(AT)echo "clean - clean all samples."
	$(AT)echo "CUDAVERSION - Specify the version of cuda you want to use, defaults to the latest installed version of cuda."

