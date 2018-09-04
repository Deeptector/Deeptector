#include <darknet.h>

network *net;

void init_net
    (
    char *cfgfile,
    char *weightfile,
    int *inw,
    int *inh,
    int *outw,
    int *outh
    )
{
    net = load_network2(cfgfile, weightfile, inw, inh, outw, outh);
    network net2; 
    set_batch_network(net, 1);
/*    *inw = net->w;
    *inh = net->h;
    *outw = net->layers[net->n - 2].out_w;
    *outh = net->layers[net->n - 2].out_h;
    printf("sizeof(net2) : %d\n", sizeof(net2));
    printf("net address : %d   size : %d\n", net, sizeof(*net));
    for(int i=0; i<107; i++) {
	layer l = net->layers[i];
       printf("%4d x%4d x%4d  %5.3f BFLOPs\n", l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
    }
    printf("net->layers[net->n - 2].out_w : %d\n", net->layers[net->n - 2].out_w);
    printf("net->layers[net->n - 2].out_h : %d\n", net->layers[net->n - 2].out_h);
    printf("\nnet -> n : %d\n", net->n);
    printf("%d %d %d %d\n", *inw, *inh, *outw, *outh);*/
}

float *run_net
    (
    float *indata
    )
{
    network_predict2(net, indata);
    return net->output;
}
