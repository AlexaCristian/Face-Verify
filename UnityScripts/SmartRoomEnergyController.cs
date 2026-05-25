using UnityEngine;
using UnityEngine.UI;

public class SmartRoomEnergyController : MonoBehaviour
{
    [Header("Room lights")]
    public Light[] controlledLights;
    public float wattsPerLight = 60f;

    [Header("UI")]
    public Text roomStatusText;
    public Text energyText;

    private int occupants;
    private bool lightsOn;
    private float energyWh;

    private void Start()
    {
        SetLights(false);
        UpdateUi();
    }

    private void Update()
    {
        if (lightsOn)
        {
            energyWh += controlledLights.Length * wattsPerLight * Time.deltaTime / 3600f;
            UpdateUi();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!other.CompareTag("Player"))
        {
            return;
        }

        occupants++;
        SetLights(true);
        UpdateUi();
    }

    private void OnTriggerExit(Collider other)
    {
        if (!other.CompareTag("Player"))
        {
            return;
        }

        occupants = Mathf.Max(0, occupants - 1);
        if (occupants == 0)
        {
            SetLights(false);
        }

        UpdateUi();
    }

    private void SetLights(bool enabled)
    {
        lightsOn = enabled;

        foreach (Light lightSource in controlledLights)
        {
            if (lightSource != null)
            {
                lightSource.enabled = enabled;
            }
        }
    }

    private void UpdateUi()
    {
        if (roomStatusText != null)
        {
            roomStatusText.text = lightsOn ? "Camera ocupata - lumini aprinse" : "Camera libera - lumini stinse";
        }

        if (energyText != null)
        {
            energyText.text = $"Consum camera: {energyWh:F2} Wh";
        }
    }
}
