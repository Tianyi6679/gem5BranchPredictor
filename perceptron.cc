#include "cpu/pred/perceptron.hh"

#include "base/bitfield.hh"
#include "base/intmath.hh"

Perceptron::Perceptron(const PerceptronParams* params)
    : BPredUnit(params),
        globalHistoryBits(ceilLog2(params->globalPredictionSize)),
        globalPredictionSize(params->globalPredictionSize),
        globalHistoryReg(params->numThreads, 0){
    if (!isPowerOf2(globalPredictionSize)){
        fatal("Invalid global predictor size! \n");
    }
    histroyRegisterMask = mask(globalHistoryBits);
    numOfPerceptron = 20;
    trainThreashold = 1.93*globalPredictionSize + 14;
    w.assign(numOfPerceptron, std::vector<unsigned>(globalPredictionSize+1 , 0));
}

void Perceptron::uncondBranch(ThreadID tid, Addr pc, void* &bp_history){
    BPHistory* history = new BPHistory;
    history->globalHistoryReg = globalHistoryReg[tid];
    history->globalTakenPred = true;
    history->globalUsed = true;
    bp_history = static_cast<void*>(history);
    //updateGlobalHistReg(tid, true);
}

void Perceptron::squash(ThreadID tid, void *bp_history) const{
    BPHistory *history = static_cast<BPHistory*>(bp_history);
    globalHistoryReg[tid] = history->globalHistoryReg;

    delete history;
}

bool Perceptron::lookup(ThreadID tid, Addr branch_addr, void* &bp_history){
    int tableIndex = branch_addr % numOfPerceptron;
    unsigned thread_history = globalHistoryReg[tid];

    int bias = w[tableIndex][0];
    int dot_product = 0;
    for (int i = 1; i<=globalPredictionSize; i++){
        // y += x_i*w_i
        if ((thread_history >> (i-1)) & 1 ){
            dot_product += w[tableIndex][i]; 
        }
        else{
            dot_product -= w[tableIndex][i]; 
        }
    }
    // if y >=0, branch is predicted to be taken
    bool taken = (dot_product+bias) >= 0 ;
    
    BPHistory *history = new BPHistory;
    history->globalHistoryReg = globalHistoryReg[tid];
    history->globalTakenPred = taken;
    bp_history = static_cast<void*>(history);
    //we only update in update func call
    //updateGlobalHistReg(tid, taken);
    return taken;
}

void Perceptron::btbUpdate(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    globalHistory[tid] &= (historyRegisterMask & ~ULL(1));
}

void Perceptron::update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history, bool squashed){
    assert(bp_history);
    
    int tableIndex = branch_addr % numOfPerceptron;
    unsigned thread_history = globalHistoryReg[tid];
    
    //training algorithm
    if (squashed || (abs(sum) <= trainThreashold)){
        if (taken) w[tableIndex][0] += 1;
        else w[tableIndex][0] += -1;
        for (int i = 1; i<=globalPredictionSize; i++){
            if (((thread_history >> (i-1)) & 1) == taken ){
                w[tableIndex][i] += 1; 
            }
            else{
                w[tableIndex][i] += -1; 
            }
        }
    }
    updateGlobalHistReg(tid, taken);
}

unsigned Perceptron::getGHR(ThreadID tid, void *bp_history) const
{
  return static_cast<BPHistory *>(bp_history)->globalHistoryReg;
}

void Perceptron::updateGlobalHistReg(ThreadID tid, bool taken){
    globalHistoryReg[tid] = taken ? (globalHistoryReg[tid] << 1) | 1 : (globalHistoryReg[tid] << 1);
    globalHistoryReg[tid] &= historyRegisterMask;
}

Perceptron* PerceptronParams::create()
{
  return new Perceptron(this);
}


