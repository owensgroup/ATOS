#common make file fragment for ufl graph datasets
#just define GRAPH_NAME prior to including this fragment

GRAPH_TAR  = $(GRAPH_NAME).tar.gz

setup: $(GRAPH_NAME).mtx

mtxfile = 0
csrfile = 0
ifeq ($(shell test -e $(GRAPH_NAME).mtx && echo -n yes),yes)
	mtxfile=1
endif

$(GRAPH_NAME).mtx: $(GRAPH_TAR)
	tar xvfz $(GRAPH_TAR)
	bash ../check.sh "$(GRAPH_NAME)"
	rm -rf $(GRAPH_NAME)

clean:
	rm $(GRAPH_NAME).mtx

realclean: clean
	rm $(GRAPH_TAR)

