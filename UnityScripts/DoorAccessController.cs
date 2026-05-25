using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class DoorAccessController : MonoBehaviour
{
    [Header("References")]
    public FaceAuthClient authClient;
    public Transform doorPivot;
    public Text statusText;

    [Header("Door movement")]
    public float openAngle = 90f;
    public float openDuration = 1.2f;
    public bool closeAutomatically = true;
    public float closeDelay = 4f;

    private bool isBusy;
    private bool isOpen;
    private Quaternion closedRotation;
    private Quaternion openRotation;

    private void Awake()
    {
        if (doorPivot == null)
        {
            doorPivot = transform;
        }

        closedRotation = doorPivot.localRotation;
        openRotation = closedRotation * Quaternion.Euler(0f, openAngle, 0f);
    }

    private async void OnTriggerEnter(Collider other)
    {
        if (!other.CompareTag("Player") || isBusy || isOpen)
        {
            return;
        }

        if (authClient == null)
        {
            authClient = FindObjectOfType<FaceAuthClient>();
        }

        if (authClient == null)
        {
            SetStatus("FaceAuthClient lipseste din scena.");
            return;
        }

        isBusy = true;
        SetStatus("Se verifica identitatea...");

        try
        {
            FaceAuthResponse response = await authClient.VerifyAsync();

            if (response.ok && response.access)
            {
                SetStatus($"Acces permis: {response.person} ({response.confidence:P0})");
                StartCoroutine(OpenDoorRoutine());
            }
            else
            {
                string reason = string.IsNullOrWhiteSpace(response.message) ? response.result : response.message;
                SetStatus($"Acces respins: {reason}");
                isBusy = false;
            }
        }
        catch (System.Exception ex)
        {
            SetStatus($"Eroare conexiune Python: {ex.Message}");
            isBusy = false;
        }
    }

    private IEnumerator OpenDoorRoutine()
    {
        yield return RotateDoor(closedRotation, openRotation);
        isOpen = true;
        isBusy = false;

        if (closeAutomatically)
        {
            yield return new WaitForSeconds(closeDelay);
            isBusy = true;
            yield return RotateDoor(openRotation, closedRotation);
            isOpen = false;
            isBusy = false;
            SetStatus("Usa inchisa.");
        }
    }

    private IEnumerator RotateDoor(Quaternion from, Quaternion to)
    {
        float elapsed = 0f;
        while (elapsed < openDuration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0f, 1f, elapsed / openDuration);
            doorPivot.localRotation = Quaternion.Slerp(from, to, t);
            yield return null;
        }

        doorPivot.localRotation = to;
    }

    private void SetStatus(string text)
    {
        Debug.Log(text);
        if (statusText != null)
        {
            statusText.text = text;
        }
    }
}
