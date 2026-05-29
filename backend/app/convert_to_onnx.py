import torch
import torch.nn as nn
import onnx

class EchoAuthentic(nn.Module):
    def __init__(self):
        super(EchoAuthentic, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, features, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_steps, channels * features)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.classifier(context_vector)
        return out.squeeze(-1)

def convert_model():
    print("Initializing ONNX Conversion...")

    model = EchoAuthentic() 
    
    model_path = "models/echo_authentic_v1.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Weights loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Could not find {model_path}. Make sure the .pth file is in the backend/models/ folder!")
        return
    
    model.eval()

    dummy_input = torch.randn(1, 1, 128, 126)

    onnx_path = "models/echo_authentic_v1.onnx"
    
    torch.onnx.export(
        model,                         
        dummy_input,                   
        onnx_path,                     
        export_params=True,            
        opset_version=14,              
        do_constant_folding=True,      
        input_names=['input_audio'],   
        output_names=['ai_probability'], 
        dynamic_axes={
            'input_audio': {0: 'batch_size', 3: 'time_steps'}, 
            'ai_probability': {0: 'batch_size'}
        }
    )
    
    print(f"Success! Highly optimized ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    convert_model()