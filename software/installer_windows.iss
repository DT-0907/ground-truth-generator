; Inno Setup Script for CCTV-YOLO Windows Installer
; Run this AFTER build_windows.bat to create a proper .exe installer
;
; Requirements: Inno Setup 6+ (https://jrsoftware.org/isinfo.php)
; Usage: iscc installer_windows.iss

[Setup]
AppName=CCTV-YOLO
AppVersion=1.0.0
AppPublisher=CCTV-YOLO
DefaultDirName={autopf}\CCTV-YOLO
DefaultGroupName=CCTV-YOLO
OutputDir=dist
OutputBaseFilename=CCTV-YOLO-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayName=CCTV-YOLO

[Files]
Source: "dist\CCTV-YOLO\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\CCTV-YOLO"; Filename: "{app}\CCTV-YOLO.exe"
Name: "{autodesktop}\CCTV-YOLO"; Filename: "{app}\CCTV-YOLO.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\CCTV-YOLO.exe"; Description: "Launch CCTV-YOLO"; Flags: nowait postinstall skipifsilent
