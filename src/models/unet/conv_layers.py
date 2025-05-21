import torch as tr
from torch import nn

class N_Conv(nn.Module):
    """([Conv] -> [BatchNorm] -> [ReLu]) x N"""

    def __init__(
        self,
        input_channels,
        output_channels,
        num_conv,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super().__init__()
        layers = []
        for i in range(num_conv):

            if i != 0:
                layers.append(
                    nn.Conv1d(
                        output_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride,
                    )
                )
            else:
                layers.append(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride,
                    )
                )
            layers.append(nn.BatchNorm1d(output_channels))
            layers.append(nn.ReLU(inplace=True))

        self.N_Conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.N_Conv(x)

class DownBlock(nn.Module):
    """
    Bloque de downsampling que combina una capa de pooling con una secuencia de convoluciones (N_Conv).

    Args:
        in_channels: Número de canales de entrada.
        out_channels: Número de canales de salida.
        num_conv: Número de capas en la secuencia N_Conv.
        pool_mode: Tipo de pooling a utilizar ("max" o "avg").
        kernel_size: Tamaño del kernel para N_Conv (por defecto 3).
        padding: Padding para N_Conv (por defecto 1).
        stride: Stride para N_Conv (por defecto 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv: int,
        pool_mode: str = "max",
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1
    ) -> None:
        super().__init__()
        if pool_mode == "max":
            pooling_layer = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        elif pool_mode == "avg":
            pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        else:
            raise ValueError('El parámetro "pool_mode" debe ser "max" o "avg".')

        self.down = nn.Sequential(
            pooling_layer,
            N_Conv(in_channels, out_channels, num_conv, kernel_size, padding, stride),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """
    Bloque de upsampling con conexión skip opcional y fusión (concatenación o suma),
    seguido de una secuencia de convoluciones (N_Conv).

    Args:
        in_channels: Canales de entrada para el upsampling.
        out_channels: Canales de salida deseados.
        num_conv: Número de capas en N_Conv.
        up_mode: Método de upsampling: "upsample" o "transpose".
        addition: Modo de fusión en la conexión skip: "cat" (concatenar) o "sum" (sumar).
        skip: Si True se utiliza la conexión skip con la entrada x2.
        kernel_size: Tamaño del kernel para N_Conv (por defecto 3).
        padding: Padding para N_Conv (por defecto 1).
        stride: Stride para N_Conv (por defecto 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv: int,
        up_mode: str = "upsample",
        skip: bool = True,
        addition: str = "cat",
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1
    ) -> None:
        super().__init__()
        self.skip = skip
        self.addition = addition
        self.up_mode = up_mode

        if up_mode not in ["upsample", "transpose"]:
            raise ValueError(
                'El parámetro "up_mode" debe ser "upsample" o "transpose".'
            )

        # Configuración del upsampling y determinación de canales tras el up.
        if up_mode == "upsample":
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            up_out_channels = in_channels  # canales permanecen iguales en upsample
        else:  # "transpose"
            self.up = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            up_out_channels = out_channels

        # Definir los canales de entrada para la siguiente convolución.
        if skip:
            if addition == "cat":
                conv_in_channels = up_out_channels + out_channels
            elif addition == "sum":
                conv_in_channels = out_channels
                if up_mode == "upsample":
                    self.adjust = nn.Conv1d(
                        up_out_channels, out_channels, kernel_size=1
                    )
            else:
                raise ValueError('El parámetro "addition" debe ser "cat" o "sum".')
        else:
            conv_in_channels = up_out_channels

        self.conv = N_Conv(
            conv_in_channels, out_channels, num_conv, kernel_size, padding, stride
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.skip:
            if x2 is None:
                raise ValueError("Se requiere x2 para la conexión skip.")
            if self.addition == "cat":
                diff = x2.size(2) - x1.size(2)
                x1 = nn.functional.pad(x1, [diff // 2, diff - diff // 2])
                x = tr.cat([x2, x1], dim=1)
            elif self.addition == "sum":
                if self.up_mode == "upsample":
                    x1 = self.adjust(x1)
                x = x2 + x1
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)