using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

[Serializable]
public class FaceAuthResponse
{
    public bool ok;
    public bool access;
    public string result;
    public string person;
    public float confidence;
    public float liveness;
    public string message;
    public string details;
    public string database;
    public float threshold;
    public string timestamp;
}

public class FaceAuthClient : MonoBehaviour
{
    [Header("Python socket server")]
    public string host = "127.0.0.1";
    public int port = 5055;
    public int timeoutMs = 15000;

    public async Task<FaceAuthResponse> VerifyAsync()
    {
        return await SendCommandAsync("VERIFY");
    }

    public async Task<FaceAuthResponse> GetStatusAsync()
    {
        return await SendCommandAsync("STATUS");
    }

    private async Task<FaceAuthResponse> SendCommandAsync(string command)
    {
        string json = await Task.Run(() => SendCommandBlocking(command));
        FaceAuthResponse response = JsonUtility.FromJson<FaceAuthResponse>(json);

        if (response == null)
        {
            throw new Exception("Python server returned an invalid response.");
        }

        return response;
    }

    private string SendCommandBlocking(string command)
    {
        using (TcpClient client = new TcpClient())
        {
            IAsyncResult connectResult = client.BeginConnect(host, port, null, null);
            bool connected = connectResult.AsyncWaitHandle.WaitOne(timeoutMs);

            if (!connected)
            {
                throw new TimeoutException($"Could not connect to Python server at {host}:{port}.");
            }

            client.EndConnect(connectResult);
            client.ReceiveTimeout = timeoutMs;
            client.SendTimeout = timeoutMs;

            using (NetworkStream stream = client.GetStream())
            using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true })
            using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
            {
                writer.WriteLine(command);
                string line = reader.ReadLine();

                if (string.IsNullOrWhiteSpace(line))
                {
                    throw new IOException("Python server closed the connection without a response.");
                }

                return line;
            }
        }
    }
}
