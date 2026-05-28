; Inno Setup Script for CCTV-YOLO v2 Windows Installer
; Built automatically at the end of build_windows.bat (which passes the
; version via /DAppVersion). You can also run it by hand:
;   iscc /DAppVersion=2.0.4 installer_windows.iss
;
; Requirements: Inno Setup 6+ (https://jrsoftware.org/isinfo.php)

; Version comes from build_windows.bat (-> __version__.py). Fall back to a
; placeholder if someone runs iscc without the define.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppName=CCTV-YOLO
AppVersion={#AppVersion}
; Stable AppId (GUID) -- DO NOT CHANGE between versions. This is the
; identity Windows uses to recognize an installed copy, so future
; installers upgrade in place and the uninstaller stays tracked.
AppId={{8F3A1C2D-5E47-4B9A-9C61-2D7E0F4A3B88}
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
; Upgrades: close a running instance, then install over the top.
CloseApplications=yes
RestartApplications=no

[Files]
Source: "dist\CCTV-YOLO\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\CCTV-YOLO"; Filename: "{app}\CCTV-YOLO.exe"
Name: "{autodesktop}\CCTV-YOLO"; Filename: "{app}\CCTV-YOLO.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\CCTV-YOLO.exe"; Description: "Launch CCTV-YOLO"; Flags: nowait postinstall skipifsilent
