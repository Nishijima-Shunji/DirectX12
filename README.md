# DirectX12

This is a small DirectX 12 sample project. The solution file `DirectX12.sln` was created with Visual Studio 2022 (v17).

## Build environment

- **Visual Studio**: tested with Visual Studio 2022 version 17.5 or newer. Make sure you install the Desktop development with C++ workload and the Windows 10/11 SDK.
- **C++ standard**: the project is set to use C++20 features.

## Dependencies

The project relies on a few external libraries which must be available on your system:

- **DirectXTK** – DirectX Tool Kit headers and libraries
- **DirectXTex** – used for texture processing
- **Assimp 5.2.5** – used to load model files

The `DirectX12.vcxproj` assumes the include directories and library paths for these libraries are located under `C:\directxtk`, `C:\DirectXTex-main`, and `C:\DirectX_lib\assimp\5.2.5` respectively. Update these paths in the project file if they differ on your machine.

## Running

1. Open `DirectX12.sln` in Visual Studio.
2. Select the `x64` configuration (`Debug` or `Release`).
3. Build the solution and run the generated executable.

The `assets` folder contains sample resources (e.g. `korosuke.fbx` and `default.png`) used by the project.

If precompiled shader objects (`*.cso`) are missing, the engine will try to
compile the corresponding HLSL files at runtime using `D3DCompileFromFile`.
Ensure that the `.cso` files exist or that the DirectX shader compiler DLLs are
available so compilation can succeed at runtime.

## License

No explicit license file is provided in this repository. Please contact the repository owner for licensing information if you wish to use the code in other projects.
