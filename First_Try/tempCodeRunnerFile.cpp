if(std::abs(prev_loss - current_loss) < 1e-6){
            std::cout<<"Early stopping at epoch: "<<epoch<<std::endl;
            break;
        }