//
//  ViewController.swift
//  Posterfiler 9000
//
//  Created by Jacob Dana on 11/20/17.
//  Copyright © 2017 Jacob Dana. All rights reserved.
//

import UIKit




class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    var newMedia: Bool?
    
    @IBAction func useCamera(_ sender: AnyObject) {
        
        if UIImagePickerController.isSourceTypeAvailable(
            UIImagePickerControllerSourceType.camera) {
            
            let imagePicker = UIImagePickerController()
            
            imagePicker.delegate = self
            imagePicker.sourceType =
                UIImagePickerControllerSourceType.camera
            imagePicker.mediaTypes = [kUTTypeImage as String]
            imagePicker.allowsEditing = false
            
            self.present(imagePicker, animated: true,
                         completion: nil)
            newMedia = true
        }
    }
    
    @IBAction func useCameraRoll(_ sender: AnyObject) {
        
        if UIImagePickerController.isSourceTypeAvailable(
            UIImagePickerControllerSourceType.savedPhotosAlbum) {
            let imagePicker = UIImagePickerController()
            
            imagePicker.delegate = self
            imagePicker.sourceType =
                UIImagePickerControllerSourceType.photoLibrary
            imagePicker.mediaTypes = [kUTTypeImage as String]
            imagePicker.allowsEditing = false
            self.present(imagePicker, animated: true,
                         completion: nil)
            newMedia = false
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}


func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
    
    let mediaType = info[UIImagePickerControllerMediaType] as! NSString
    
    self.dismiss(animated: true, completion: nil)
    
    if mediaType.isEqual(to: UTTypeImage as String) {
        let image = info[UIImagePickerControllerOriginalImage]
            as! UIImage
        
        imageView.image = image
        
        if (newMedia == true) {
            UIImageWriteToSavedPhotosAlbum(image, self,
                                           #selector(ViewController.image(image:didFinishSavingWithError:contextInfo:)), nil)
        } else if mediaType.isEqual(to: kUTTypeMovie as String) {
            // Code to support video here
        }
        
    }
}

func image(image: UIImage, didFinishSavingWithError error: NSErrorPointer, contextInfo:UnsafeRawPointer) {
    
    if error != nil {
        let alert = UIAlertController(title: "Save Failed",
                                      message: "Failed to save image",
                                      preferredStyle: UIAlertControllerStyle.alert)
        
        let cancelAction = UIAlertAction(title: "OK",
                                         style: .cancel, handler: nil)
        
        alert.addAction(cancelAction)
        self.present(alert, animated: true,
                     completion: nil)
    }
}

